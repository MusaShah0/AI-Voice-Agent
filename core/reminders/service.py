from django.db import transaction
from django.conf import settings
from django.utils import timezone

from core.models import Appointment, Reminder, ReminderEvent

from .policy import first_fire_at, minutes_left, snooze_decision


TERMINAL_STATUSES = (
    Reminder.Status.ACKNOWLEDGED,
    Reminder.Status.DONE,
    Reminder.Status.EXPIRED,
    Reminder.Status.CANCELLED,
)


def record_event(reminder, event_type, payload=None):
    ReminderEvent.objects.create(
        reminder=reminder,
        event_type=event_type,
        payload=payload or {},
    )


class ReminderService:
    @staticmethod
    @transaction.atomic
    def create_for_appointment(appointment, now=None):
        now = now or timezone.now()
        next_fire_at = first_fire_at(appointment.start_time, now)
        if next_fire_at is None:
            return None

        reminder, created = Reminder.objects.get_or_create(
            appointment=appointment,
            defaults={
                "next_fire_at": next_fire_at,
            },
        )
        if not created and reminder.status in Reminder.OPEN_STATUSES:
            reminder.next_fire_at = next_fire_at
            reminder.status = Reminder.Status.SCHEDULED
            reminder.attempt_count = 0
            reminder.snooze_count = 0
            reminder.save(update_fields=[
                "next_fire_at", "status", "attempt_count", "snooze_count", "updated_at",
            ])
        record_event(
            reminder,
            "created" if created else "reset",
            {"next_fire_at": next_fire_at.isoformat()},
        )
        return reminder

    @staticmethod
    @transaction.atomic
    def reset_for_appointment(appointment_id, now=None):
        appointment = Appointment.objects.get(pk=appointment_id)
        now = now or timezone.now()
        next_fire_at = first_fire_at(appointment.start_time, now)
        reminder = Reminder.objects.filter(appointment=appointment).first()
        if next_fire_at is None:
            if reminder and reminder.status in Reminder.OPEN_STATUSES:
                reminder.status = Reminder.Status.EXPIRED
                reminder.next_fire_at = None
                reminder.save(update_fields=["status", "next_fire_at", "updated_at"])
                record_event(reminder, "skipped_too_imminent")
            return None
        if reminder is None:
            return ReminderService.create_for_appointment(appointment, now=now)
        reminder.status = Reminder.Status.SCHEDULED
        reminder.next_fire_at = next_fire_at
        reminder.attempt_count = 0
        reminder.snooze_count = 0
        reminder.acknowledged_at = None
        reminder.save(update_fields=[
            "status", "next_fire_at", "attempt_count", "snooze_count",
            "acknowledged_at", "updated_at",
        ])
        record_event(reminder, "rescheduled", {"next_fire_at": next_fire_at.isoformat()})
        return reminder

    @staticmethod
    @transaction.atomic
    def cancel_for_appointment(appointment_id):
        reminder = Reminder.objects.filter(appointment_id=appointment_id).first()
        if reminder and reminder.status in Reminder.OPEN_STATUSES:
            reminder.status = Reminder.Status.CANCELLED
            reminder.next_fire_at = None
            reminder.save(update_fields=["status", "next_fire_at", "updated_at"])
            record_event(reminder, "cancelled")

    @staticmethod
    @transaction.atomic
    def record_call_started(reminder_id, now=None):
        now = now or timezone.now()
        reminder = Reminder.objects.select_related("appointment").get(pk=reminder_id)
        if reminder.status in TERMINAL_STATUSES:
            return reminder
        reminder.call_count += 1
        reminder.last_call_at = now
        reminder.save(update_fields=["call_count", "last_call_at", "updated_at"])
        record_event(reminder, "call_started")
        return reminder

    @staticmethod
    def get_call_context(reminder_id, now=None):
        now = now or timezone.now()
        reminder = Reminder.objects.select_related(
            "appointment",
            "appointment__doctor",
        ).get(pk=reminder_id)
        appointment = reminder.appointment
        decision = snooze_decision(now, appointment.start_time)
        return {
            "reminder_id": reminder.pk,
            "appointment_id": appointment.pk,
            "patient_name": appointment.caller_name,
            "purpose": appointment.purpose,
            "start_time": appointment.start_time,
            "clinic_tz": appointment.doctor.timezone,
            "doctor_name": appointment.doctor.name,
            "now": now,
            "minutes_left": minutes_left(now, appointment.start_time),
            "reminder_status": reminder.status,
            "snooze_allowed": decision.allowed,
            "snooze_until": decision.until,
            "snooze_denied_reason": None if decision.allowed else decision.reason,
        }

    @staticmethod
    @transaction.atomic
    def acknowledge(reminder_id, now=None):
        now = now or timezone.now()
        reminder = Reminder.objects.select_for_update().get(pk=reminder_id)
        if reminder.status in TERMINAL_STATUSES:
            return {"ok": True, "status": reminder.status, "message": "That reminder is already handled."}
        reminder.status = Reminder.Status.ACKNOWLEDGED
        reminder.next_fire_at = None
        reminder.acknowledged_at = now
        reminder.save(update_fields=["status", "next_fire_at", "acknowledged_at", "updated_at"])
        record_event(reminder, "acknowledged")
        return {"ok": True, "status": reminder.status}

    @staticmethod
    @transaction.atomic
    def snooze(reminder_id, now=None, minutes=None):
        now = now or timezone.now()
        reminder = Reminder.objects.select_related("appointment").get(pk=reminder_id)
        if reminder.status in TERMINAL_STATUSES:
            return {"ok": False, "status": reminder.status, "reason": "reminder_closed"}
        decision = snooze_decision(now, reminder.appointment.start_time, minutes)
        if not decision.allowed:
            record_event(reminder, "snooze_denied", {"reason": decision.reason})
            return {
                "ok": False,
                "reason": decision.reason,
                "minutes": minutes if minutes is not None else settings.SNOOZE_MINUTES,
                "minutes_left": minutes_left(now, reminder.appointment.start_time),
            }
        reminder.status = Reminder.Status.SNOOZED
        reminder.next_fire_at = decision.until
        reminder.attempt_count = 0
        reminder.snooze_count += 1
        reminder.save(update_fields=[
            "status", "next_fire_at", "attempt_count", "snooze_count", "updated_at",
        ])
        duration_minutes = minutes if minutes is not None else settings.SNOOZE_MINUTES
        record_event(
            reminder,
            "snoozed",
            {"until": decision.until.isoformat(), "minutes": duration_minutes},
        )
        return {
            "ok": True,
            "status": reminder.status,
            "until": decision.until,
            "minutes": duration_minutes,
        }
