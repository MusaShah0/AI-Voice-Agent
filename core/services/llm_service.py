import datetime
import json
import logging
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq

from core.models import Doctor, Appointment
from core.repositories import DataRepository
from core.services.calendar_service import GoogleCalendarService, CalendarAuthError

logger = logging.getLogger(__name__)


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
                "description": "Check whether Dr. Melvin is available for a date and time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "string",
                            "description": "Appointment start datetime in ISO 8601 format.",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "Appointment end datetime in ISO 8601 format.",
                        },
                    },
                    "required": ["start_time", "end_time"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "book_appointment",
                "description": "Book an appointment after the caller confirms the time, name, and purpose.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "caller_name": {
                            "type": "string",
                            "description": "Full name of the caller or patient.",
                        },
                        "start_time": {
                            "type": "string",
                            "description": "Confirmed appointment start datetime in ISO 8601 format.",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "Confirmed appointment end datetime in ISO 8601 format.",
                        },
                        "purpose": {
                            "type": "string",
                            "description": "Reason for the appointment.",
                        },
                    },
                    "required": ["caller_name", "start_time", "end_time", "purpose"],
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

    def _get_system_instruction(self) -> str:
        now_dt = datetime.datetime.now()
        now_str = now_dt.strftime("%A, %B %d, %Y at %I:%M %p")
        return (
            "You are Janus, an AI front-desk medical receptionist for Dr. Melvin's clinic. "
            f"The current date and time is {now_str}. "
            "Your duties:\n"
            "1. Help callers check Dr. Melvin's availability and book appointments.\n"
            "2. When the caller asks about a specific date or time, call check_doctor_availability.\n"
            "3. If the slot is available, confirm with the caller and request their name and visit purpose if not yet provided.\n"
            "4. If the slot has a conflict (is busy), politely explain the conflict and suggest the alternative open time.\n"
            "5. Once the caller confirms their name, visit purpose, and preferred time slot, call book_appointment.\n"
            "6. Always keep your verbal responses concise, warm, natural, and conversational for voice output.\n"
            "7. After booking, say the appointment has been added to Dr. Melvin's clinic schedule. "
            "The connected Google Calendar belongs to the clinic, not the caller. Never claim that the "
            "appointment was added to the caller's personal calendar unless a tool confirms that.\n"
            "8. Never output markdown asterisks, bullet lists, or formatting symbols that would sound unnatural when read aloud."
        )

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
                dt = datetime.datetime.now() + datetime.timedelta(days=1)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt

    @_log_tool_timing("check_doctor_availability")
    def check_doctor_availability(self, start_time: str, end_time: str) -> Dict[str, Any]:
        """Checks whether Dr. Melvin is available or has a schedule conflict for a given time window.

        Args:
            start_time: Start date and time of the proposed slot (ISO format or YYYY-MM-DD HH:MM:SS)
            end_time: End date and time of the proposed slot (ISO format or YYYY-MM-DD HH:MM:SS)
        """
        logger.info("Tool check_doctor_availability called")
        try:
            start_dt = self._parse_datetime(start_time)
            end_dt = self._parse_datetime(end_time)
        except Exception as parse_err:
            logger.warning("Error parsing availability query times: %s", parse_err)
            return {
                "available": False,
                "conflicts": [],
                "message": "I could not determine the exact date and time. Please ask the caller to clarify."
            }

        # 1. Check Google Calendar if authenticated
        try:
            if self.calendar_service.is_authenticated():
                result = self.calendar_service.check_availability(start_dt, end_dt)
                if not result.get("available"):
                    self.last_tool_status = "conflict_detected"
                    return {
                        "available": False,
                        "conflicts": result.get("conflicts", []),
                        "message": "Doctor has an existing event on Google Calendar during this time."
                    }
        except Exception as cal_err:
            logger.warning("Google Calendar availability query failed: %s", cal_err)

        # 2. Check Local Database for conflicts
        try:
            conflicting_apps = Appointment.objects.filter(
                status="Scheduled",
                start_time__lt=end_dt,
                end_time__gt=start_dt
            )
            if conflicting_apps.exists():
                conflict = conflicting_apps.first()
                self.last_tool_status = "conflict_detected"
                return {
                    "available": False,
                    "conflicts": [{"summary": f"Appointment: {conflict.purpose}"}],
                    "message": f"Dr. Melvin is busy with an existing appointment from {conflict.start_time.strftime('%I:%M %p')} to {conflict.end_time.strftime('%I:%M %p')}."
                }
        except Exception as db_err:
            logger.error("Database conflict check failed: %s", db_err)

        self.last_tool_status = "availability_checked"
        return {
            "available": True,
            "conflicts": [],
            "message": f"Dr. Melvin is completely free between {start_dt.strftime('%I:%M %p')} and {end_dt.strftime('%I:%M %p')}."
        }

    @_log_tool_timing("book_appointment")
    def book_appointment(
        self,
        caller_name: str,
        start_time: str,
        end_time: str,
        purpose: str
    ) -> Dict[str, Any]:
        """Books and confirms an appointment with Dr. Melvin in Google Calendar and the database.

        Args:
            caller_name: Full name of the caller/patient
            start_time: Start date and time of the confirmed appointment
            end_time: End date and time of the confirmed appointment
            purpose: Reason or symptoms for the appointment
        """
        logger.info("Tool book_appointment called")
        try:
            start_dt = self._parse_datetime(start_time)
            end_dt = self._parse_datetime(end_time)
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
                purpose=purpose
            )

            # 3. Create follow-up chart review task for Dr. Melvin
            try:
                DataRepository.create_task_sync(
                    appointment_id=appointment_id,
                    description=f"Review medical chart for appointment with {caller_name} ({purpose})",
                    due_time=start_dt - datetime.timedelta(hours=1)
                )
            except Exception as task_err:
                logger.warning("Could not create preparation task: %s", task_err)

            self.last_tool_status = "calendar_booked"
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
        messages.append({"role": "user", "content": user_input})
        return messages

    @staticmethod
    def _serialize_assistant_message(message) -> Dict[str, Any]:
        serialized = {"role": "assistant", "content": message.content or ""}
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            serialized["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]
        return serialized

    def process_dialog(
        self,
        conversation_history: list,
        user_input: str
    ) -> Tuple[str, Optional[str]]:
        """
        Manages dialog turn, executes LLM tool calling (availability check and booking),
        and returns the natural assistant text reply and tool execution status.
        """
        self.last_tool_status = None
        user_input = (user_input or "").strip()
        if not user_input:
            return "I am listening. How can I help you today?", None

        if self.client is None:
            logger.error("Groq client is uninitialized (missing GROQ_API_KEY or client initialization failure).")
            return "I am having trouble connecting to my scheduling system right now. Could you please repeat that?", "error"

        response_text = ""
        messages = self._build_messages(conversation_history, user_input)
        tool_handlers = {
            "check_doctor_availability": self.check_doctor_availability,
            "book_appointment": self.book_appointment,
        }

        try:
            for request_number in range(self.MAX_TOOL_CALL_ROUNDS + 1):
                request_started_at = time.perf_counter()
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=self.TOOL_DEFINITIONS,
                        tool_choice="auto",
                        max_completion_tokens=300,
                    )
                except Exception:
                    logger.exception(
                        "Pipeline Groq: request failed model=%s elapsed_ms=%.2f request=%s",
                        self.model_name,
                        (time.perf_counter() - request_started_at) * 1000,
                        request_number + 1,
                    )
                    raise

                logger.info(
                    "Pipeline Groq: response received model=%s elapsed_ms=%.2f request=%s",
                    self.model_name,
                    (time.perf_counter() - request_started_at) * 1000,
                    request_number + 1,
                )
                message = response.choices[0].message
                tool_calls = getattr(message, "tool_calls", None) or []
                if not tool_calls:
                    response_text = (message.content or "").strip()
                    if not response_text:
                        raise ValueError("Groq response had neither message text nor tool calls")
                    break

                if request_number == self.MAX_TOOL_CALL_ROUNDS:
                    raise RuntimeError("Groq exceeded the maximum number of tool-call rounds")

                messages.append(self._serialize_assistant_message(message))
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    handler = tool_handlers.get(tool_name)
                    if handler is None:
                        result = {"status": "error", "message": "Unknown scheduling tool."}
                        self.last_tool_status = "error"
                    else:
                        try:
                            arguments = json.loads(tool_call.function.arguments or "{}")
                        except (TypeError, json.JSONDecodeError):
                            result = {"status": "error", "message": "Tool arguments were not valid JSON."}
                            self.last_tool_status = "error"
                        else:
                            if not isinstance(arguments, dict):
                                result = {"status": "error", "message": "Tool arguments must be a JSON object."}
                                self.last_tool_status = "error"
                            else:
                                logger.info("Pipeline Groq: executing tool=%s", tool_name)
                                result = handler(**arguments)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result, default=str),
                    })
            else:
                raise RuntimeError("Groq tool-call loop ended without a final response")
        except Exception as err:
            logger.exception("Groq dialog failed model=%s error=%s", self.model_name, err)
            response_text = "I am having trouble connecting to my scheduling system right now. Could you please repeat that?"
            self.last_tool_status = "error"

        # Update outer conversation history
        try:
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response_text})
        except Exception as hist_err:
            logger.warning("Failed to update conversation history: %s", hist_err)

        tool_triggered = self.last_tool_status
        return response_text, tool_triggered