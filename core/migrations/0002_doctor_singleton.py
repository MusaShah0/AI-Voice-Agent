from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.AddConstraint(
            model_name='doctor',
            constraint=models.CheckConstraint(
                condition=models.Q(id=1),
                name='doctor_singleton_id',
            ),
        ),
    ]