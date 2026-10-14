import logging
from datetime import timedelta
from zoneinfo import ZoneInfo

from django.conf import settings
from django.core.mail import send_mail
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Appointment, Doctor

logger = logging.getLogger(__name__)
REMINDER_LEAD_TIME = timedelta(minutes=15)


class Command(BaseCommand):
    help = "Email minimal appointment reminders 15 minutes before scheduled visits."

    def handle(self, *args, **options):
        now = timezone.now()
        doctor = Doctor.objects.get(pk=Doctor.SINGLETON_ID)
        doctor_timezone = ZoneInfo(doctor.timezone)
        appointments = Appointment.objects.filter(
            doctor=doctor,
            status="Scheduled",
            start_time__gt=now,
            start_time__lte=now + REMINDER_LEAD_TIME,
            reminder_sent_at__isnull=True,
        ).order_by("start_time", "pk")
        sent_count = 0
        failures = []

        for appointment in appointments:
            local_start = timezone.localtime(appointment.start_time, doctor_timezone)
            subject = f"Appointment in 15 minutes: {appointment.caller_name}"
            body = (
                f"Patient: {appointment.caller_name}\n"
                f"Appointment time: {local_start:%A, %B %d, %Y at %I:%M %p %Z}\n"
            )
            try:
                delivered = send_mail(
                    subject=subject,
                    message=body,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[doctor.email],
                    fail_silently=False,
                )
                if delivered != 1:
                    raise RuntimeError("Email backend did not confirm delivery.")
            except Exception as exc:
                failures.append(appointment.pk)
                logger.exception(
                    "Appointment reminder email failed appointment_id=%s recipient=%s",
                    appointment.pk,
                    doctor.email,
                )
                self.stderr.write(
                    f"Reminder failed for appointment {appointment.pk}: {exc}"
                )
                continue

            updated = Appointment.objects.filter(
                pk=appointment.pk,
                status="Scheduled",
                reminder_sent_at__isnull=True,
            ).update(reminder_sent_at=timezone.now())
            if updated:
                sent_count += 1
                self.stdout.write(f"Reminder sent for appointment {appointment.pk}.")
            else:
                logger.warning(
                    "Appointment reminder delivered but appointment changed before marking sent "
                    "appointment_id=%s",
                    appointment.pk,
                )

        self.stdout.write(f"Sent {sent_count} appointment reminder(s).")
        if failures:
            raise CommandError(
                f"Failed to send reminders for appointment IDs: {', '.join(map(str, failures))}"
            )
