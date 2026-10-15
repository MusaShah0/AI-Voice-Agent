from django.db import models

SINGLETON_DOCTOR_ID = 1

class Doctor(models.Model):
    SINGLETON_ID = SINGLETON_DOCTOR_ID

    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    timezone = models.CharField(max_length=50, default='Asia/Karachi')
    dashboard_last_seen_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=models.Q(id=SINGLETON_DOCTOR_ID),
                name='doctor_singleton_id',
            ),
        ]

class Appointment(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    caller_name = models.CharField(max_length=255)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    purpose = models.TextField()
    status = models.CharField(max_length=50, default='Scheduled')
    google_event_id = models.CharField(max_length=255, null=True, blank=True)


class Reminder(models.Model):
    class Status(models.TextChoices):
        SCHEDULED = "scheduled", "Scheduled"
        ACTIVE = "active", "Active"
        SNOOZED = "snoozed", "Snoozed"
        ACKNOWLEDGED = "acknowledged", "Acknowledged"
        DONE = "done", "Done"
        EXPIRED = "expired", "Expired"
        CANCELLED = "cancelled", "Cancelled"

    OPEN_STATUSES = (Status.SCHEDULED, Status.ACTIVE, Status.SNOOZED)

    appointment = models.OneToOneField(
        Appointment,
        on_delete=models.CASCADE,
        related_name="reminder",
    )
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.SCHEDULED)
    next_fire_at = models.DateTimeField(null=True, blank=True)
    attempt_count = models.PositiveSmallIntegerField(default=0)
    snooze_count = models.PositiveSmallIntegerField(default=0)
    call_count = models.PositiveSmallIntegerField(default=0)
    last_fired_at = models.DateTimeField(null=True, blank=True)
    last_call_at = models.DateTimeField(null=True, blank=True)
    last_channel = models.CharField(max_length=20, blank=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [models.Index(fields=["status", "next_fire_at"], name="core_reminder_due_idx")]


class ReminderEvent(models.Model):
    reminder = models.ForeignKey(Reminder, on_delete=models.CASCADE, related_name="events")
    event_type = models.CharField(max_length=40)
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Task(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    appointment = models.ForeignKey(Appointment, on_delete=models.SET_NULL, null=True)
    description = models.TextField()
    due_time = models.DateTimeField()
    status = models.CharField(max_length=20, default='Pending') # Pending, Done, Snoozed

class Transcript(models.Model):
    appointment = models.ForeignKey(Appointment, on_delete=models.SET_NULL, null=True, blank=True)
    session_id = models.CharField(max_length=100)
    full_transcript = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)