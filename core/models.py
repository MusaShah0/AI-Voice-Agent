from django.db import models

SINGLETON_DOCTOR_ID = 1

class Doctor(models.Model):
    SINGLETON_ID = SINGLETON_DOCTOR_ID

    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    timezone = models.CharField(max_length=50, default='Asia/Karachi')

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
    reminder_sent_at = models.DateTimeField(null=True, blank=True)

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