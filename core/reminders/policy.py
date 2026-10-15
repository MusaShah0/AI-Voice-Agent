from dataclasses import dataclass
from datetime import datetime, timedelta

from django.conf import settings


@dataclass(frozen=True)
class SnoozeDecision:
    allowed: bool
    until: datetime | None
    reason: str


def first_fire_at(start: datetime, now: datetime) -> datetime | None:
    minimum_lead = timedelta(minutes=settings.REMINDER_MIN_LEAD_TO_REMIND_MINUTES)
    if start - now < minimum_lead:
        return None
    return max(
        start - timedelta(minutes=settings.REMINDER_LEAD_MINUTES),
        now + timedelta(seconds=30),
    )


def snooze_decision(
    now: datetime,
    start: datetime,
    minutes: int | None = None,
) -> SnoozeDecision:
    duration_minutes = settings.SNOOZE_MINUTES if minutes is None else minutes
    if duration_minutes < 1:
        return SnoozeDecision(False, None, "invalid_duration")

    until = now + timedelta(minutes=duration_minutes)
    latest_allowed = start - timedelta(minutes=settings.SNOOZE_MIN_BUFFER_MINUTES)
    if until <= latest_allowed:
        return SnoozeDecision(True, until, "ok")
    return SnoozeDecision(False, None, "too_close_to_start")


def minutes_left(now: datetime, start: datetime) -> int:
    return int((start - now).total_seconds() // 60)
