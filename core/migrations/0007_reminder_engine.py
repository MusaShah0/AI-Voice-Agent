from datetime import timedelta

from django.db import migrations, models
from django.utils import timezone
import django.db.models.deletion


def create_existing_reminders(apps, schema_editor):
    Appointment = apps.get_model("core", "Appointment")
    Task = apps.get_model("core", "Task")
    Reminder = apps.get_model("core", "Reminder")
    database = schema_editor.connection.alias
    now = timezone.now()
    minimum_start = now + timedelta(minutes=2)
    appointments = list(
        Appointment.objects.using(database).filter(
            status="Scheduled",
            start_time__gte=minimum_start,
        ).values("id", "start_time")
    )
    appointment_ids = [appointment["id"] for appointment in appointments]
    tasks_by_appointment = {}
    for task in Task.objects.using(database).filter(
        appointment_id__in=appointment_ids,
        status="Pending",
    ).order_by("pk").values("id", "appointment_id"):
        tasks_by_appointment.setdefault(task["appointment_id"], task["id"])

    Reminder.objects.using(database).bulk_create([
        Reminder(
            appointment_id=appointment["id"],
            task_id=tasks_by_appointment.get(appointment["id"]),
            status="scheduled",
            next_fire_at=max(
                appointment["start_time"] - timedelta(minutes=30),
                now + timedelta(seconds=30),
            ),
        )
        for appointment in appointments
    ], batch_size=500)


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0006_appointment_reminder_sent_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="doctor",
            name="dashboard_last_seen_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.CreateModel(
            name="Reminder",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("status", models.CharField(choices=[
                    ("scheduled", "Scheduled"),
                    ("active", "Active"),
                    ("snoozed", "Snoozed"),
                    ("acknowledged", "Acknowledged"),
                    ("done", "Done"),
                    ("expired", "Expired"),
                    ("cancelled", "Cancelled"),
                ], default="scheduled", max_length=20)),
                ("next_fire_at", models.DateTimeField(blank=True, null=True)),
                ("attempt_count", models.PositiveSmallIntegerField(default=0)),
                ("snooze_count", models.PositiveSmallIntegerField(default=0)),
                ("call_count", models.PositiveSmallIntegerField(default=0)),
                ("last_fired_at", models.DateTimeField(blank=True, null=True)),
                ("last_call_at", models.DateTimeField(blank=True, null=True)),
                ("last_channel", models.CharField(blank=True, max_length=20)),
                ("acknowledged_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("appointment", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="reminder", to="core.appointment")),
                ("task", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="core.task")),
            ],
        ),
        migrations.CreateModel(
            name="ReminderEvent",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("event_type", models.CharField(max_length=40)),
                ("payload", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("reminder", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="events", to="core.reminder")),
            ],
        ),
        migrations.AddIndex(
            model_name="reminder",
            index=models.Index(fields=["status", "next_fire_at"], name="core_reminder_due_idx"),
        ),
        migrations.RunPython(create_existing_reminders, migrations.RunPython.noop),
    ]
