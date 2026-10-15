import time

from django.core.management.base import BaseCommand

from core.reminders.sweep import sweep_reminders


class Command(BaseCommand):
    help = "Run the in-app appointment reminder scheduler."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="Run one sweep and exit.")

    def handle(self, *args, **options):
        if options["once"]:
            fired = sweep_reminders()
            self.stdout.write(f"Reminder sweep complete; {fired} reminder(s) activated.")
            return

        self.stdout.write("Reminder scheduler running (30-second interval).")
        while True:
            fired = sweep_reminders()
            if fired:
                self.stdout.write(f"Activated {fired} in-app reminder(s).")
            time.sleep(30)
