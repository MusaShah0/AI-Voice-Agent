from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0005_appointment_google_event_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='appointment',
            name='reminder_sent_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
