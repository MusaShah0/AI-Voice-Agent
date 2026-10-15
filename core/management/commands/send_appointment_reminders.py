from django.core.management.base import BaseCommand

from core.reminders.sweep import sweep_reminders


class Command(BaseCommand):
    help = "Compatibility alias for one in-app appointment reminder sweep."

    def handle(self, *args, **options):
        activated = sweep_reminders()
        self.stdout.write(f"Reminder sweep complete; {activated} reminder(s) activated.")
