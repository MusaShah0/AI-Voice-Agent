from datetime import timedelta
from zoneinfo import ZoneInfo

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_GET

from .models import Appointment, Doctor

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