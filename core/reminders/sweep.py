from datetime import timedelta

from django.conf import settings
from django.db.models import F
from django.utils import timezone

from core.models import Appointment, Reminder, ReminderEvent


def sweep_reminders(now=None):
    now = now or timezone.now()
    due_ids = Reminder.objects.filter(
        status__in=Reminder.OPEN_STATUSES,
        next_fire_at__lte=now,
    ).values_list("pk", flat=True)
    fired = 0
    for reminder_id in due_ids.iterator():
        if claim_and_fire(reminder_id, now):
            fired += 1
    return fired


def claim_and_fire(reminder_id, now=None):
    now = now or timezone.now()
    reminder = Reminder.objects.select_related("appointment").filter(pk=reminder_id).first()
    if (
        reminder is None
        or reminder.next_fire_at is None
        or reminder.next_fire_at > now
    ):
        return False

    appointment = reminder.appointment
    if appointment.status != "Scheduled":
        _terminate(reminder, Reminder.Status.CANCELLED, "cancelled")
        return False
    if now >= appointment.start_time or reminder.attempt_count >= settings.REMINDER_MAX_ATTEMPTS:
        _terminate(reminder, Reminder.Status.EXPIRED, "expired")
        return False

    next_fire = min(
        now + timedelta(minutes=settings.REMINDER_REPEAT_MINUTES),
        appointment.start_time,
    )
    claimed = Reminder.objects.filter(
        pk=reminder.pk,
        next_fire_at=reminder.next_fire_at,
        status__in=Reminder.OPEN_STATUSES,
    ).update(
        status=Reminder.Status.ACTIVE,
        next_fire_at=next_fire,
        attempt_count=F("attempt_count") + 1,
        last_fired_at=now,
        last_channel="in_app",
    )
    if claimed != 1:
        return False

    ReminderEvent.objects.create(
        reminder=reminder,
        event_type="fired",
        payload={"channel": "in_app", "attempt": reminder.attempt_count + 1},
    )
    return True


def _terminate(reminder, status, event_type):
    changed = Reminder.objects.filter(
        pk=reminder.pk,
        status__in=Reminder.OPEN_STATUSES,
        next_fire_at=reminder.next_fire_at,
    ).update(status=status, next_fire_at=None)
    if changed:
        ReminderEvent.objects.create(reminder=reminder, event_type=event_type)
