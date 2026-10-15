import datetime
import logging
from zoneinfo import ZoneInfo

from django.utils import timezone

from core.models import Appointment, Doctor
from core.repositories import DataRepository
from core.services.calendar_service import GoogleCalendarService

logger = logging.getLogger(__name__)
DEFAULT_APPOINTMENT_DURATION = datetime.timedelta(minutes=30)


class AppointmentService:
    def __init__(self, calendar_service=None):
        self.calendar_service = calendar_service or GoogleCalendarService()

    @staticmethod
    def _parse_datetime(value, clinic_timezone):
        parsed = datetime.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if timezone.is_naive(parsed):
            parsed = parsed.replace(tzinfo=ZoneInfo(clinic_timezone))
        return parsed

    def check_availability(self, start, end=None, exclude_appointment_id=None):
        doctor = Doctor.objects.get(pk=Doctor.SINGLETON_ID)
        start_time = self._parse_datetime(start, doctor.timezone)
        end_time = (
            self._parse_datetime(end, doctor.timezone)
            if end
            else start_time + DEFAULT_APPOINTMENT_DURATION
        )
        if end_time <= start_time:
            return {"ok": False, "message": "The end time must be after the start time."}

        conflicts = Appointment.objects.filter(
            doctor=doctor,
            status="Scheduled",
            start_time__lt=end_time,
            end_time__gt=start_time,
        )
        if exclude_appointment_id is not None:
            conflicts = conflicts.exclude(pk=exclude_appointment_id)
        conflict = conflicts.order_by("start_time").first()
        if conflict:
            return {
                "ok": False,
                "conflict": True,
                "message": "That time conflicts with another scheduled appointment.",
            }
        return {"ok": True, "start_time": start_time, "end_time": end_time}

    def get_today_schedule(self, now=None):
        now = now or timezone.now()
        doctor = Doctor.objects.get(pk=Doctor.SINGLETON_ID)
        local_now = timezone.localtime(now, ZoneInfo(doctor.timezone))
        day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + datetime.timedelta(days=1)
        rows = Appointment.objects.filter(
            doctor=doctor,
            status="Scheduled",
            start_time__gte=max(day_start, local_now),
            start_time__lt=day_end,
        ).order_by("start_time")
        return [
            {
                "patient_name": row.caller_name,
                "start_time": timezone.localtime(row.start_time, ZoneInfo(doctor.timezone)),
                "end_time": timezone.localtime(row.end_time, ZoneInfo(doctor.timezone)),
            }
            for row in rows
        ]

    def reschedule(self, appointment_id, new_start, new_end=None):
        try:
            appointment = Appointment.objects.get(
                pk=appointment_id,
                status="Scheduled",
            )
        except Appointment.DoesNotExist:
            return {"ok": False, "status": "not_found", "message": "That appointment is no longer scheduled."}

        doctor = appointment.doctor
        try:
            start_time = self._parse_datetime(new_start, doctor.timezone)
            end_time = (
                self._parse_datetime(new_end, doctor.timezone)
                if new_end
                else start_time + (appointment.end_time - appointment.start_time)
            )
        except (TypeError, ValueError):
            return {"ok": False, "status": "invalid_time", "message": "I couldn't understand that appointment time."}
        if start_time <= timezone.now():
            return {"ok": False, "status": "invalid_time", "message": "The new appointment time must be in the future."}
        if end_time <= start_time:
            return {"ok": False, "status": "invalid_time", "message": "The appointment end time must be after its start time."}

        availability = self.check_availability(
            start_time,
            end_time,
            exclude_appointment_id=appointment.pk,
        )
        if not availability["ok"]:
            return {"ok": False, "status": "conflict", "message": availability["message"]}

        calendar_synced = False
        if self.calendar_service.is_authenticated():
            if not appointment.google_event_id:
                return {
                    "ok": False,
                    "status": "error",
                    "message": "This appointment has no linked calendar event, so it was not changed.",
                }
            try:
                self.calendar_service.update_event(
                    event_id=appointment.google_event_id,
                    start_time=start_time,
                    end_time=end_time,
                )
                calendar_synced = True
            except Exception:
                logger.exception("Doctor reminder reschedule calendar update failed appointment_id=%s", appointment.pk)
                return {
                    "ok": False,
                    "status": "error",
                    "message": "The calendar could not be updated, so the appointment was not changed.",
                }

        try:
            result = DataRepository.update_appointment_sync(
                appointment_id=appointment.pk,
                start_time=start_time,
                end_time=end_time,
            )
            if result["status"] != "updated":
                raise RuntimeError("The appointment could not be updated in the database.")
        except Exception:
            logger.exception("Doctor reminder reschedule database update failed appointment_id=%s", appointment.pk)
            if calendar_synced:
                try:
                    self.calendar_service.update_event(
                        event_id=appointment.google_event_id,
                        start_time=appointment.start_time,
                        end_time=appointment.end_time,
                    )
                except Exception:
                    logger.exception("Doctor reminder reschedule calendar rollback failed appointment_id=%s", appointment.pk)
                    return {
                        "ok": False,
                        "status": "error",
                        "message": "The calendar changed but the database did not, and the calendar rollback failed. Please reconcile this appointment.",
                    }
            return {
                "ok": False,
                "status": "error",
                "message": "The appointment could not be updated in the scheduling database.",
            }

        return {
            "ok": True,
            "status": "rescheduled",
            "start_time": start_time,
            "end_time": end_time,
            "calendar_synced": calendar_synced,
            "patient_name": appointment.caller_name,
        }
