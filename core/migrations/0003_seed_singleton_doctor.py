from django.db import migrations


def seed_doctor(apps, schema_editor):
    Doctor = apps.get_model('core', 'Doctor')
    Doctor.objects.get_or_create(
        id=1,
        defaults={
            'name': 'Dr. Smith',
            'email': 'dr.smith@example.com',
            'timezone': 'UTC',
        }
    )


def reverse_seed(apps, schema_editor):
    Doctor = apps.get_model('core', 'Doctor')
    Doctor.objects.filter(id=1).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_doctor_singleton'),
    ]

    operations = [
        migrations.RunPython(seed_doctor, reverse_seed),
    ]
