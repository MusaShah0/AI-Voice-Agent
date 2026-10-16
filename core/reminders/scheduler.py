import logging
import threading
import time

from django.db import DatabaseError, connections
from django.db import close_old_connections

from core.reminders.sweep import sweep_reminders


logger = logging.getLogger("core.reminders.scheduler")
_scheduler_thread = None
_scheduler_lock = threading.Lock()
_SCHEDULER_INTERVAL_SECONDS = 30


def start_reminder_scheduler():
    global _scheduler_thread

    with _scheduler_lock:
        if _scheduler_thread is not None and _scheduler_thread.is_alive():
            return

        _scheduler_thread = threading.Thread(
            target=_run_scheduler,
            name="appointment-reminder-scheduler",
            daemon=True,
        )
        _scheduler_thread.start()
        logger.info("In-process reminder scheduler started.")


def _run_scheduler():
    while True:
        close_old_connections()
        try:
            fired = sweep_reminders()
            if fired:
                logger.info("Activated %s in-app reminder(s).", fired)
        except DatabaseError:
            logger.exception("Reminder sweep failed; it will retry on the next interval.")
        finally:
            connections.close_all()

        time.sleep(_SCHEDULER_INTERVAL_SECONDS)
