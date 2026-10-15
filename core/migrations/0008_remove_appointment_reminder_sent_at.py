from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0007_reminder_engine"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="appointment",
            name="reminder_sent_at",
        ),
    ]
