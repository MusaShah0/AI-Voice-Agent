from datetime import timedelta
from zoneinfo import ZoneInfo

from django.conf import settings
from django.core import signing
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .models import Appointment, Doctor, Reminder

CALL_TOKEN_SALT = "core.reminders.call"

def index_view(request):
    return HttpResponse("Voice Scheduling Agent Backend is running successfully.")


@require_GET
def doctor_dashboard_data(request):
    if not settings.DEBUG:
        return JsonResponse({"error": "Doctor dashboard is available only in development."}, status=404)

    now = timezone.now()
    doctor = Doctor.objects.get(pk=Doctor.SINGLETON_ID)
    doctor_timezone = ZoneInfo(doctor.timezone)
    appointments = Appointment.objects.filter(
        doctor=doctor,
    ).order_by("-start_time")
    upcoming = appointments.filter(
        status="Scheduled",
        start_time__gte=now,
    ).order_by("start_time")
    reminder_window = upcoming.filter(
        start_time__lte=now + timedelta(minutes=15),
    )

    def serialize(appointment):
        return {
            "id": appointment.pk,
            "patient_name": appointment.caller_name,
            "start_time": timezone.localtime(appointment.start_time, doctor_timezone).isoformat(),
            "end_time": timezone.localtime(appointment.end_time, doctor_timezone).isoformat(),
            "purpose": appointment.purpose,
            "status": appointment.status,
        }

    return JsonResponse({
        "generated_at": timezone.localtime(now, doctor_timezone).isoformat(),
        "doctor": {
            "name": doctor.name,
            "timezone": doctor.timezone,
        },
        "upcoming": [serialize(appointment) for appointment in upcoming],
        "all_appointments": [serialize(appointment) for appointment in appointments],
        "reminders": [serialize(appointment) for appointment in reminder_window],
    })


@require_GET
def active_reminders(request):
    if not settings.DEBUG:
        return JsonResponse({"error": "Doctor reminders are available only in development."}, status=404)

    now = timezone.now()
    doctor = Doctor.objects.get(pk=Doctor.SINGLETON_ID)
    Doctor.objects.filter(pk=doctor.pk).update(dashboard_last_seen_at=now)
    doctor_timezone = ZoneInfo(doctor.timezone)
    reminders = Reminder.objects.select_related("appointment").filter(
        appointment__doctor=doctor,
        appointment__status="Scheduled",
        appointment__start_time__gt=now,
        status__in=(Reminder.Status.ACTIVE, Reminder.Status.SNOOZED),
    ).order_by("appointment__start_time")
    active = []
    for reminder in reminders:
        start_time = timezone.localtime(reminder.appointment.start_time, doctor_timezone)
        active.append({
            "reminder_id": reminder.pk,
            "appointment_id": reminder.appointment_id,
            "patient_name": reminder.appointment.caller_name,
            "start_time": start_time.isoformat(),
            "minutes_until_start": int((reminder.appointment.start_time - now).total_seconds() // 60),
            "attempt_count": reminder.attempt_count,
            "status": reminder.status,
            "next_fire_at": (
                timezone.localtime(reminder.next_fire_at, doctor_timezone).isoformat()
                if reminder.status == Reminder.Status.SNOOZED and reminder.next_fire_at
                else None
            ),
        })
    return JsonResponse({
        "server_time": timezone.localtime(now, doctor_timezone).isoformat(),
        "active": active,
    })


@csrf_exempt
@require_POST
def reminder_call_token(request, reminder_id):
    if not settings.DEBUG:
        return JsonResponse({"error": "Doctor calls are available only in development."}, status=404)

    now = timezone.now()
    reminder = Reminder.objects.select_related("appointment").filter(pk=reminder_id).first()
    if reminder is None:
        return JsonResponse({"error": "Reminder not found."}, status=404)
    if (
        reminder.status not in Reminder.OPEN_STATUSES
        or reminder.appointment.status != "Scheduled"
        or reminder.appointment.start_time <= now
    ):
        return JsonResponse({"error": "This reminder is no longer available."}, status=409)

    token = signing.TimestampSigner(salt=CALL_TOKEN_SALT).sign_object(
        {"reminder_id": reminder.pk}
    )
    websocket_scheme = "wss" if request.is_secure() else "ws"
    host = request.get_host()
    return JsonResponse({
        "token": token,
        "ws_url": f"{websocket_scheme}://{host}/ws/doc-voice/?token={token}",
    })