import datetime
import json
import logging
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from django.conf import settings
from django.utils import timezone
from django.utils import timezone as django_timezone
from groq import Groq

from core.models import Doctor, Appointment
from core.repositories import DataRepository
from core.services.calendar_service import GoogleCalendarService, CalendarAuthError

logger = logging.getLogger(__name__)
CLINIC_TIMEZONE = ZoneInfo(settings.TIME_ZONE)
DEFAULT_APPOINTMENT_DURATION = datetime.timedelta(minutes=30)


def _log_tool_timing(tool_name):
    def decorate(tool):
        @wraps(tool)
        def wrapped(*args, **kwargs):
            started_at = time.perf_counter()
            try:
                return tool(*args, **kwargs)
            finally:
                logger.info(
                    "Pipeline LLM tool: completed tool=%s elapsed_ms=%.2f",
                    tool_name,
                    (time.perf_counter() - started_at) * 1000,
                )

        return wrapped

    return decorate


class LLMService:
    """
    Manages dialog state and Groq LLM interactions with tool calling
    for checking doctor availability and booking appointments.
    """

    DEFAULT_MODEL = "openai/gpt-oss-20b"
    MAX_TOOL_CALL_ROUNDS = 5
    TOOL_DEFINITIONS = [
        {
            "type": "function",
            "function": {
                "name": "check_doctor_availability",
                "description": "Check whether Dr. Melvin is available for a date and time. Before calling this tool, compare the requested start time with the current time; if it is not strictly in the future, do not check availability and tell the caller the appointment time has passed. If the caller gives only a start time, check a 30-minute appointment window.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "string",
                            "description": "Appointment start datetime in the clinic's local timezone (Asia/Karachi, UTC+05:00), ISO 8601 format including +05:00 offset.",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "Appointment end datetime in the clinic's local timezone (Asia/Karachi, UTC+05:00), ISO 8601 format including +05:00 offset.",
                        },
                        "caller_name": {
                            "type": "string",
                            "description": "Patient name if already provided for this booking; omit if not yet known.",
                        },
                        "purpose": {
                            "type": "string",
                            "description": "Visit purpose if already provided for this booking; omit if not yet known.",
                        },
                    },
                    "required": ["start_time"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "book_appointment",
                "description": "Book a 30-minute appointment when the caller has requested a booking and the required name, start time, and purpose are known, and the start time is strictly in the future. Do not book a past or current time. If the caller explicitly requests a different end time or duration, pass that end time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "caller_name": {
                            "type": "string",
                            "description": "Full name of the caller or patient.",
                        },
                        "start_time": {
                            "type": "string",
                            "description": "Confirmed appointment start datetime in the clinic's local timezone (Asia/Karachi, UTC+05:00), ISO 8601 format including +05:00 offset.",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "Confirmed appointment end datetime in the clinic's local timezone (Asia/Karachi, UTC+05:00), ISO 8601 format including +05:00 offset.",
                        },
                        "purpose": {
                            "type": "string",
                            "description": "Reason for the appointment.",
                        },
                    },
                    "required": ["start_time"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_upcoming_appointments",
                "description": "Look up a caller's upcoming scheduled appointments by their name before discussing or rescheduling them.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "caller_name": {
                            "type": "string",
                            "description": "Patient name as provided by the caller.",
                        },
                    },
                    "required": ["caller_name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reschedule_appointment",
                "description": "Move one upcoming appointment after matching the caller's name and existing appointment start time against the database. If new_end_time is omitted, preserve the original duration; if the caller requests a specific end time, use it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "caller_name": {
                            "type": "string",
                            "description": "Patient name as provided by the caller.",
                        },
                        "existing_start_time": {
                            "type": "string",
                            "description": "Exact date and start time of the existing appointment in clinic-local time with +05:00 offset.",
                        },
                        "new_start_time": {
                            "type": "string",
                            "description": "Requested new start date and time in clinic local time with +05:00 offset.",
                        },
                        "new_end_time": {
                            "type": "string",
                            "description": "Optional explicitly requested new end date and time in clinic local time with +05:00 offset. Omit to preserve the original appointment duration.",
                        },
                    },
                    "required": ["caller_name", "existing_start_time", "new_start_time"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_appointment",
                "description": "Cancel a specific upcoming appointment after matching the caller's name and existing appointment start time against the database. Do not cancel if the match is missing or ambiguous.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "caller_name": {
                            "type": "string",
                            "description": "Patient name as provided by the caller.",
                        },
                        "existing_start_time": {
                            "type": "string",
                            "description": "Exact date and start time of the appointment to cancel in clinic-local time with +05:00 offset.",
                        },
                    },
                    "required": ["caller_name", "existing_start_time"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    def __init__(self, calendar_service: Optional[GoogleCalendarService] = None):
        api_key = os.environ.get("GROQ_API_KEY")
        self.model_name = os.environ.get("GROQ_MODEL", self.DEFAULT_MODEL).strip()
        if not self.model_name:
            self.model_name = self.DEFAULT_MODEL
        try:
            self.client = Groq(api_key=api_key) if api_key else None
        except Exception as client_err:
            logger.exception("Failed to initialize Groq client: %s", client_err)
            self.client = None

        self.calendar_service = calendar_service or GoogleCalendarService()
        self.last_tool_status = None
        self.booking_context: Dict[str, str] = {}

    def _get_system_instruction(self) -> str:
        now_dt = django_timezone.localtime(django_timezone.now(), CLINIC_TIMEZONE)
        now_str = now_dt.strftime("%A, %B %d, %Y at %I:%M %p")
        return (
            "You are Janus, an AI front-desk medical receptionist for Dr. Melvin's clinic. "
            f"The current date and time is {now_str}. "
            "Interpret all caller dates and times in the clinic's local timezone, Asia/Karachi (UTC+05:00). "
            "Pass appointment times to scheduling tools with an explicit +05:00 offset, and speak all times "
            "back to the caller in that same clinic-local timezone. Never reinterpret a clinic-local time as UTC. "
            "Your duties:\n"
            "1. Help callers check Dr. Melvin's availability and book appointments.\n"
            "2. Before using any scheduling tool, compare the requested start time with the current date and time above "
            "in the clinic timezone. If it is not strictly in the future, do not call availability or booking tools; "
            "tell the caller that time has passed and ask for a future time. New appointments are 30 minutes by default. "
            "When the caller asks to book at a specific start time, "
            "call check_doctor_availability for that window and include every booking detail already stated, especially "
            "caller_name and purpose. If only a start time is given, check a 30-minute window; use a different duration "
            "only if explicitly requested.\n"
            "3. Keep supplied booking details from every caller turn. After availability is confirmed, book using the "
            "details already supplied and ask only for required details that are still missing. Never ask again for "
            "a name, purpose, date, or time the caller already provided. If only the name is missing, ask only for the name.\n"
            "4. If the slot has a conflict (is busy), politely explain the conflict and suggest the alternative open time.\n"
            "5. When the caller has requested a booking and their name, visit purpose, and preferred time are known, call book_appointment without asking them to repeat or reconfirm supplied details.\n"
            "6. When a caller asks to hear their upcoming schedule, ask for their name if it is missing, "
            "then call list_upcoming_appointments before giving appointment details. If several appointments are returned, "
            "read their dates and times and ask which appointment they mean when unclear. "
            "For cancellation, ask for the caller's name or appointment date/time if either is missing. "
            "When both are clear, call cancel_appointment directly; it verifies the patient and exact appointment "
            "against the database, and refuses missing or ambiguous matches. Confirm cancellation only if its result "
            "is cancelled.\n"
            "For a reschedule, ask for the caller's name, the existing appointment date/time, or the new date/time if any "
            "are missing. When all three are clear, call reschedule_appointment directly; that tool verifies the name and "
            "existing appointment time in the database and rejects missing or ambiguous matches. This avoids an unnecessary "
            "extra model/tool round trip. If the caller specifies both a new start and end time, pass new_end_time; otherwise "
            "omit it and preserve the appointment's existing duration. "
            "Do not claim the change succeeded unless that tool returns status updated.\n"
            "7. Always keep your verbal responses concise, warm, natural, and conversational for voice output.\n"
            "8. After booking, say the appointment has been added to Dr. Melvin's clinic schedule. "
            "The connected Google Calendar belongs to the clinic, not the caller. Never claim that the "
            "appointment was added to the caller's personal calendar unless a tool confirms that.\n"
            "9. After completing a booking, cancellation, reschedule, or appointment lookup, ask briefly, "
            "'Is there anything else I can help you with?'. If the caller says no or that is all, the system "
            "will thank them and end the call.\n"
            "10. Never output markdown asterisks, bullet lists, or formatting symbols that would sound unnatural when read aloud."
        )

    def _remember_booking_details(
        self,
        caller_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> None:
        for key, value in (
            ("caller_name", caller_name),
            ("start_time", start_time),
            ("end_time", end_time),
            ("purpose", purpose),
        ):
            if isinstance(value, str) and value.strip():
                self.booking_context[key] = value.strip()

    def _parse_datetime(self, dt_str: str) -> datetime.datetime:
        """Helper to parse varied date/time formats returned by LLM into a timezone-aware datetime."""
        dt_str = str(dt_str).strip().replace("Z", "+00:00")
        try:
            dt = datetime.datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M",
                "%Y-%m-%d",
            ):
                try:
                    dt = datetime.datetime.strptime(dt_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Unsupported date/time value: {dt_str!r}")

        if django_timezone.is_naive(dt):
            dt = dt.replace(tzinfo=CLINIC_TIMEZONE)
        return dt.astimezone(CLINIC_TIMEZONE)

    @_log_tool_timing("check_doctor_availability")
    def check_doctor_availability(
        self,
        start_time: str,
        end_time: Optional[str] = None,
        caller_name: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Checks whether Dr. Melvin is available or has a schedule conflict for a given time window.

        Args:
            start_time: Start date and time of the proposed slot (ISO format or YYYY-MM-DD HH:MM:SS)
            end_time: Optional end date/time; defaults to 30 minutes after start_time.
        """
        logger.info("Tool check_doctor_availability called")
        try:
            start_dt = self._parse_datetime(start_time)
            if start_dt <= timezone.now():
                self.booking_context.clear()
                return {
                    "ok": False,
                    "available": False,
                    "status": "invalid_time",
                    "message": "That appointment time has passed. Please choose a future time.",
                }
            end_dt = (
                self._parse_datetime(end_time)
                if end_time
                else start_dt + DEFAULT_APPOINTMENT_DURATION
            )
            if end_dt <= start_dt:
                raise ValueError("Availability window end must be later than its start.")
            self.booking_context.clear()
            self._remember_booking_details(
                caller_name=caller_name,
                start_time=start_dt.isoformat(),
                end_time=end_dt.isoformat(),
                purpose=purpose,
            )
        except Exception as parse_err:
            logger.warning("Error parsing availability query times: %s", parse_err)
            return {
                "available": False,
                "conflicts": [],
                "message": "I could not determine the exact date and time. Please ask the caller to clarify."
            }

        # SQLite is the source of truth for appointment conflicts; Google is only used when booking.
        try:
            conflict = Appointment.objects.filter(
                status="Scheduled",
                start_time__lt=end_dt,
                end_time__gt=start_dt
            ).first()
            if conflict:
                self.last_tool_status = "conflict_detected"
                return {
                    "available": False,
                    "conflicts": [{"summary": f"Appointment: {conflict.purpose}"}],
                    "message": (
                        "Dr. Melvin is busy with an existing appointment from "
                        f"{django_timezone.localtime(conflict.start_time, CLINIC_TIMEZONE).strftime('%I:%M %p')} "
                        f"to {django_timezone.localtime(conflict.end_time, CLINIC_TIMEZONE).strftime('%I:%M %p')}."
                    ),
                    "booking_details": dict(self.booking_context),
                }
        except Exception as db_err:
            logger.exception("Database conflict check failed: %s", db_err)
            self.last_tool_status = "error"
            return {
                "available": False,
                "conflicts": [],
                "message": "Availability could not be verified due to a database error.",
            }

        self.last_tool_status = "availability_checked"
        return {
            "available": True,
            "conflicts": [],
            "booking_details": dict(self.booking_context),
            "message": (
                "Dr. Melvin is completely free between "
                f"{start_dt.strftime('%I:%M %p')} and {end_dt.strftime('%I:%M %p')}."
            )
        }

    @_log_tool_timing("list_upcoming_appointments")
    def list_upcoming_appointments(self, caller_name: str) -> Dict[str, Any]:
        """Returns up to ten future scheduled appointments matching the patient's name."""
        normalized_name = caller_name.strip()
        if not normalized_name:
            self.last_tool_status = "error"
            return {"status": "error", "message": "A patient name is required."}

        appointments = DataRepository.list_upcoming_appointments_sync(
            normalized_name,
            django_timezone.now(),
        )
        if not appointments:
            self.last_tool_status = "appointments_not_found"
            return {
                "status": "not_found",
                "appointments": [],
                "message": "No upcoming scheduled appointments were found for that name.",
            }

        self.last_tool_status = "appointments_found"
        return {
            "status": "found",
            "appointments": [
                {
                    "appointment_id": appointment["id"],
                    "start_time": appointment["start_time"].astimezone(CLINIC_TIMEZONE).isoformat(),
                    "end_time": appointment["end_time"].astimezone(CLINIC_TIMEZONE).isoformat(),
                    "start_display": appointment["start_time"].astimezone(CLINIC_TIMEZONE).strftime(
                        "%A, %B %d at %I:%M %p"
                    ),
                    "end_display": appointment["end_time"].astimezone(CLINIC_TIMEZONE).strftime(
                        "%I:%M %p"
                    ),
                    "purpose": appointment["purpose"],
                }
                for appointment in appointments
            ],
            "message": f"Found {len(appointments)} upcoming scheduled appointment(s).",
        }

    @_log_tool_timing("reschedule_appointment")
    def reschedule_appointment(
        self,
        caller_name: str,
        existing_start_time: str,
        new_start_time: str,
        new_end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reschedules a uniquely matched future appointment and syncs Google Calendar."""
        normalized_name = caller_name.strip()
        if not normalized_name:
            self.last_tool_status = "error"
            return {"status": "error", "message": "A patient name is required."}

        try:
            old_start = self._parse_datetime(existing_start_time)
            new_start = self._parse_datetime(new_start_time)
            requested_end = self._parse_datetime(new_end_time) if new_end_time else None
        except (TypeError, ValueError) as parse_err:
            self.last_tool_status = "error"
            return {
                "status": "error",
                "message": f"Could not understand the appointment date or time: {parse_err}",
            }

        now = django_timezone.now()
        if new_start <= now:
            self.last_tool_status = "error"
            return {
                "status": "error",
                "message": "The new appointment time must be in the future.",
            }

        old_minute = old_start.replace(second=0, microsecond=0)
        matches = list(Appointment.objects.filter(
            caller_name__iexact=normalized_name,
            status="Scheduled",
            start_time__gte=max(old_minute, now),
            start_time__lt=old_minute + datetime.timedelta(minutes=1),
        ).order_by("pk")[:2])

        if not matches:
            self.last_tool_status = "appointment_not_found"
            return {
                "status": "not_found",
                "message": (
                    "No upcoming scheduled appointment matched that name and time. "
                    "Please check the patient's schedule and clarify which appointment to move."
                ),
            }
        if len(matches) > 1:
            self.last_tool_status = "appointment_ambiguous"
            return {
                "status": "ambiguous",
                "message": (
                    "More than one appointment matched that name and time. "
                    "Ask the caller to clarify which appointment they mean."
                ),
            }

        appointment = matches[0]
        duration = appointment.end_time - appointment.start_time
        if duration <= datetime.timedelta(0):
            self.last_tool_status = "error"
            return {"status": "error", "message": "The existing appointment has an invalid duration."}

        new_end = requested_end or new_start + duration
        if new_end <= new_start:
            self.last_tool_status = "error"
            return {
                "status": "error",
                "message": "The new appointment end time must be later than its start time.",
            }
        if Appointment.objects.filter(
            status="Scheduled",
            start_time__lt=new_end,
            end_time__gt=new_start,
        ).exclude(pk=appointment.pk).exists():
            self.last_tool_status = "conflict_detected"
            return {
                "status": "conflict",
                "message": "That new time conflicts with another scheduled appointment.",
            }

        calendar_synced = False
        if self.calendar_service.is_authenticated():
            if not appointment.google_event_id:
                self.last_tool_status = "error"
                return {
                    "status": "error",
                    "message": (
                        "This appointment has no linked clinic calendar event, so it was not changed. "
                        "Please contact the clinic to reschedule it."
                    ),
                }
            try:
                self.calendar_service.update_event(
                    event_id=appointment.google_event_id,
                    start_time=new_start,
                    end_time=new_end,
                )
                calendar_synced = True
            except Exception:
                logger.exception(
                    "Google Calendar update failed appointment_id=%s",
                    appointment.pk,
                )
                self.last_tool_status = "error"
                return {
                    "status": "error",
                    "message": (
                        "The clinic calendar could not be updated, so the appointment was not changed."
                    ),
                }

        try:
            update_result = DataRepository.update_appointment_sync(
                appointment_id=appointment.pk,
                start_time=new_start,
                end_time=new_end,
            )
            if update_result["status"] == "conflict":
                raise RuntimeError("The requested appointment time now conflicts with another appointment.")
        except Exception:
            if calendar_synced:
                try:
                    self.calendar_service.update_event(
                        event_id=appointment.google_event_id,
                        start_time=appointment.start_time,
                        end_time=appointment.end_time,
                    )
                except Exception:
                    logger.exception(
                        "Google Calendar rollback failed appointment_id=%s",
                        appointment.pk,
                    )
                    self.last_tool_status = "error"
                    return {
                        "status": "error",
                        "message": (
                            "The clinic calendar changed, but the database update failed and the "
                            "calendar rollback also failed. Please contact the clinic."
                        ),
                    }
            logger.exception("Appointment database update failed appointment_id=%s", appointment.pk)
            self.last_tool_status = "error"
            return {
                "status": "error",
                "message": "The appointment could not be updated in the scheduling database.",
            }

        self.last_tool_status = "appointment_rescheduled"
        return {
            "status": "updated",
            "appointment_id": appointment.pk,
            "caller_name": appointment.caller_name,
            "start_time": new_start.strftime("%A, %B %d at %I:%M %p"),
            "end_time": new_end.strftime("%I:%M %p"),
            "google_calendar_synced": calendar_synced,
            "message": "The appointment was rescheduled successfully.",
        }

    @_log_tool_timing("cancel_appointment")
    def cancel_appointment(
        self,
        caller_name: str,
        existing_start_time: str,
    ) -> Dict[str, Any]:
        """Cancels a uniquely matched upcoming appointment in the DB and linked calendar."""
        normalized_name = caller_name.strip()
        if not normalized_name:
            self.last_tool_status = "error"
            return {"status": "error", "message": "A patient name is required."}

        try:
            old_start = self._parse_datetime(existing_start_time)
        except (TypeError, ValueError) as parse_err:
            self.last_tool_status = "error"
            return {
                "status": "error",
                "message": f"Could not understand the appointment date or time: {parse_err}",
            }

        now = django_timezone.now()
        old_minute = old_start.replace(second=0, microsecond=0)
        matches = list(Appointment.objects.filter(
            caller_name__iexact=normalized_name,
            status="Scheduled",
            start_time__gte=max(old_minute, now),
            start_time__lt=old_minute + datetime.timedelta(minutes=1),
        ).order_by("pk")[:2])

        if not matches:
            self.last_tool_status = "appointment_not_found"
            return {
                "status": "not_found",
                "message": "No upcoming scheduled appointment matched that name and time.",
            }
        if len(matches) > 1:
            self.last_tool_status = "appointment_ambiguous"
            return {
                "status": "ambiguous",
                "message": "More than one appointment matched that name and time. Ask which one to cancel.",
            }

        appointment = matches[0]
        calendar_synced = False
        if self.calendar_service.is_authenticated():
            if not appointment.google_event_id:
                self.last_tool_status = "error"
                return {
                    "status": "error",
                    "message": (
                        "This appointment has no linked clinic calendar event, so it was not cancelled. "
                        "Please contact the clinic."
                    ),
                }
            try:
                self.calendar_service.delete_event(event_id=appointment.google_event_id)
                calendar_synced = True
            except Exception:
                logger.exception(
                    "Google Calendar cancellation failed appointment_id=%s",
                    appointment.pk,
                )
                self.last_tool_status = "error"
                return {
                    "status": "error",
                    "message": "The clinic calendar could not cancel this event, so the appointment was not changed.",
                }

        try:
            result = DataRepository.cancel_appointment_sync(appointment.pk)
            if result["status"] != "cancelled":
                raise RuntimeError("Appointment was not cancelled in the database.")
        except Exception:
            if calendar_synced:
                logger.exception(
                    "Appointment database cancellation failed after calendar event deletion appointment_id=%s",
                    appointment.pk,
                )
                self.last_tool_status = "error"
                return {
                    "status": "error",
                    "message": (
                        "The clinic calendar event was deleted, but the database cancellation failed. "
                        "Please contact the clinic to reconcile this appointment."
                    ),
                }
            logger.exception("Appointment database cancellation failed appointment_id=%s", appointment.pk)
            self.last_tool_status = "error"
            return {
                "status": "error",
                "message": "The appointment could not be cancelled in the scheduling database.",
            }

        self.last_tool_status = "appointment_cancelled"
        return {
            "status": "cancelled",
            "appointment_id": appointment.pk,
            "caller_name": appointment.caller_name,
            "start_time": appointment.start_time.astimezone(CLINIC_TIMEZONE).strftime(
                "%A, %B %d at %I:%M %p"
            ),
            "google_calendar_synced": calendar_synced,
            "message": "The appointment was cancelled.",
        }

    @_log_tool_timing("book_appointment")
    def book_appointment(
        self,
        caller_name: str,
        start_time: str,
        purpose: str,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Books and confirms an appointment with Dr. Melvin in Google Calendar and the database.

        Args:
            caller_name: Full name of the caller/patient
            start_time: Start date and time of the confirmed appointment
            end_time: Optional end date/time; defaults to 30 minutes after the start.
            purpose: Reason or symptoms for the appointment
        """
        logger.info("Tool book_appointment called")
        try:
            start_dt = self._parse_datetime(start_time)
            if start_dt <= timezone.now():
                self.booking_context.clear()
                self.last_tool_status = "invalid_time"
                return {
                    "status": "invalid_time",
                    "message": "The appointment start time must be in the future.",
                }
            end_dt = (
                self._parse_datetime(end_time)
                if end_time
                else start_dt + DEFAULT_APPOINTMENT_DURATION
            )
            if end_dt <= start_dt:
                raise ValueError("Appointment end time must be later than its start time.")
        except Exception as parse_err:
            logger.warning("Error parsing appointment slot: %s", parse_err)
            return {
                "status": "error",
                "message": "Could not understand the date or time for booking."
            }

        google_event_id = None

        # 1. Sync to Google Calendar if authenticated
        try:
            if self.calendar_service.is_authenticated():
                event_data = self.calendar_service.create_event(
                    summary=f"Clinical Appointment: {caller_name}",
                    start_time=start_dt,
                    end_time=end_dt,
                    description=f"Purpose: {purpose}\nCaller: {caller_name}",
                    caller_name=caller_name
                )
                google_event_id = event_data.get("id")
        except Exception as cal_err:
            logger.warning("Failed to add event to Google Calendar: %s", cal_err)

        # 2. Persist Appointment to Database
        try:
            appointment_id = DataRepository.create_appointment_sync(
                caller_name=caller_name,
                start_time=start_dt,
                end_time=end_dt,
                purpose=purpose,
                google_event_id=google_event_id,
            )

            self.last_tool_status = "calendar_booked"
            self.booking_context.clear()
            return {
                "status": "confirmed",
                "appointment_id": appointment_id,
                "caller_name": caller_name,
                "start_time": start_dt.strftime("%A, %B %d at %I:%M %p"),
                "end_time": end_dt.strftime("%I:%M %p"),
                "purpose": purpose,
                "google_calendar_synced": bool(google_event_id)
            }
        except Exception as db_err:
            logger.exception("Database error while booking appointment: %s", db_err)
            self.last_tool_status = "error"
            return {
                "status": "error",
                "message": f"Could not record appointment in system: {db_err}"
            }

    def _build_messages(self, conversation_history: list, user_input: str) -> List[Dict[str, Any]]:
        """Builds the provider message list from prior turns and the current user input."""
        messages = [{"role": "system", "content": self._get_system_instruction()}]
        for turn in conversation_history:
            try:
                role = turn.get("role")
                content = turn.get("content", "")
                if role in ("user", "assistant") and isinstance(content, str) and content:
                    messages.append({"role": role, "content": content})
            except Exception as parse_err:
                logger.warning("Skipping malformed conversation-history turn: %s", parse_err)
        if self.booking_context:
            messages.append({
                "role": "system",
                "content": (
                    "The current booking request has these structured details already extracted "
                    "from the caller's messages, encoded as JSON: "
                    f"{json.dumps(self.booking_context, ensure_ascii=False)}. "
                    "Reuse them. Do not ask the caller to repeat any field that is present."
                ),
            })
        messages.append({"role": "user", "content": user_input})
        return messages

    def process_dialog(
        self,
        conversation_history: list,
        user_input: str
    ) -> Tuple[str, Optional[str]]:
        response_text = "".join(
            self.stream_dialog(conversation_history, user_input)
        )
        return response_text, self.last_tool_status

    def stream_dialog(self, conversation_history: list, user_input: str):
        """
        Streams assistant text while executing any requested scheduling tools first.
        """
        self.last_tool_status = None
        user_input = (user_input or "").strip()
        if not user_input:
            yield "I am listening. How can I help you today?"
            return

        if self.client is None:
            logger.error("Groq client is uninitialized (missing GROQ_API_KEY or client initialization failure).")
            self.last_tool_status = "error"
            yield "I am having trouble connecting to my scheduling system right now. Could you please repeat that?"
            return

        response_text = []
        completed = False
        messages = self._build_messages(conversation_history, user_input)
        tool_handlers = {
            "check_doctor_availability": self.check_doctor_availability,
            "book_appointment": self.book_appointment,
            "list_upcoming_appointments": self.list_upcoming_appointments,
            "reschedule_appointment": self.reschedule_appointment,
            "cancel_appointment": self.cancel_appointment,
        }

        try:
            for request_number in range(self.MAX_TOOL_CALL_ROUNDS + 1):
                request_started_at = time.perf_counter()
                try:
                    stream = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=self.TOOL_DEFINITIONS,
                        tool_choice="auto",
                        max_completion_tokens=280,
                        reasoning_effort="low",
                        stream=True,
                    )
                except Exception:
                    logger.exception(
                        "Pipeline Groq: request failed model=%s elapsed_ms=%.2f request=%s",
                        self.model_name,
                        (time.perf_counter() - request_started_at) * 1000,
                        request_number + 1,
                    )
                    raise

                tool_call_fragments = {}
                try:
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None)
                        if content:
                            response_text.append(content)
                            yield content

                        for tool_delta in getattr(delta, "tool_calls", None) or []:
                            tool_fragment = tool_call_fragments.setdefault(
                                tool_delta.index,
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tool_delta.id:
                                tool_fragment["id"] = tool_delta.id
                            function_delta = tool_delta.function
                            if function_delta.name:
                                tool_fragment["function"]["name"] += function_delta.name
                            if function_delta.arguments:
                                tool_fragment["function"]["arguments"] += function_delta.arguments
                finally:
                    stream.close()

                logger.info(
                    "Pipeline Groq: response stream completed model=%s elapsed_ms=%.2f request=%s",
                    self.model_name,
                    (time.perf_counter() - request_started_at) * 1000,
                    request_number + 1,
                )
                tool_calls = [
                    tool_call_fragments[index]
                    for index in sorted(tool_call_fragments)
                ]
                if not tool_calls:
                    if not "".join(response_text).strip():
                        raise ValueError("Groq response had neither message text nor tool calls")
                    completed = True
                    break

                if response_text:
                    raise ValueError("Groq streamed text and tool calls in the same response")

                if request_number == self.MAX_TOOL_CALL_ROUNDS:
                    raise RuntimeError("Groq exceeded the maximum number of tool-call rounds")

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                })
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    handler = tool_handlers.get(tool_name)
                    if handler is None:
                        result = {"status": "error", "message": "Unknown scheduling tool."}
                        self.last_tool_status = "error"
                    else:
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"] or "{}")
                        except (TypeError, json.JSONDecodeError):
                            result = {"status": "error", "message": "Tool arguments were not valid JSON."}
                            self.last_tool_status = "error"
                        else:
                            if not isinstance(arguments, dict):
                                result = {"status": "error", "message": "Tool arguments must be a JSON object."}
                                self.last_tool_status = "error"
                            else:
                                logger.info("Pipeline Groq: executing tool=%s", tool_name)
                                try:
                                    if tool_name == "book_appointment":
                                        arguments = {
                                            **self.booking_context,
                                            **arguments,
                                        }
                                        missing_fields = [
                                            field
                                            for field in (
                                                "caller_name",
                                                "start_time",
                                                "purpose",
                                            )
                                            if not str(arguments.get(field, "")).strip()
                                        ]
                                        if missing_fields:
                                            self.last_tool_status = "booking_details_required"
                                            result = {
                                                "status": "missing_details",
                                                "missing_fields": missing_fields,
                                                "message": (
                                                    "Booking was not created. Ask only for these "
                                                    "missing details: "
                                                    + ", ".join(missing_fields)
                                                ),
                                            }
                                        else:
                                            result = handler(**arguments)
                                    else:
                                        result = handler(**arguments)
                                except Exception as tool_err:
                                    logger.exception("Groq tool execution failed tool=%s", tool_name)
                                    result = {"status": "error", "message": str(tool_err)}
                                    self.last_tool_status = "error"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "content": json.dumps(result, default=str),
                    })
            else:
                raise RuntimeError("Groq tool-call loop ended without a final response")
        except Exception as err:
            logger.exception("Groq dialog failed model=%s error=%s", self.model_name, err)
            error_reply = "I am having trouble connecting to my scheduling system right now. Could you please repeat that?"
            if response_text:
                error_reply = " I am having trouble completing that response. Please ask me to repeat it."
            response_text.append(error_reply)
            yield error_reply
            self.last_tool_status = "error"

        if completed or self.last_tool_status == "error":
            try:
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({
                    "role": "assistant",
                    "content": "".join(response_text),
                })
            except Exception as hist_err:
                logger.warning("Failed to update conversation history: %s", hist_err)