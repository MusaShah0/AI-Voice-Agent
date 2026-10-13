import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from core.models import Doctor, Appointment
from core.repositories import DataRepository
from core.services.calendar_service import GoogleCalendarService, CalendarAuthError

logger = logging.getLogger(__name__)


class LLMService:
    """
    Manages dialog state and Gemini LLM interactions with tool calling
    for checking doctor availability and booking appointments.
    """

    CANDIDATE_MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
    ]

    def __init__(self, calendar_service: Optional[GoogleCalendarService] = None):
        api_key = os.environ.get("GEMINI_API_KEY")
        try:
            self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        except Exception as client_err:
            logger.exception("Failed to initialize Google GenAI Client: %s", client_err)
            self.client = None

        self.model_name = self.CANDIDATE_MODELS[0]
        self.fallback_models = self.CANDIDATE_MODELS[1:]
        self.calendar_service = calendar_service or GoogleCalendarService()
        self.last_tool_status = None
        self._chat = None

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
            "7. Never output markdown asterisks, bullet lists, or formatting symbols that would sound unnatural when read aloud."
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

    def _get_tools(self) -> List[Any]:
        return [self.check_doctor_availability, self.book_appointment]

    def _build_config(self) -> types.GenerateContentConfig:
        """Constructs safe generation config without invalid thinking parameters."""
        try:
            tools = self._get_tools()
        except Exception as e:
            logger.warning("Failed to resolve tools: %s", e)
            tools = []

        return types.GenerateContentConfig(
            system_instruction=self._get_system_instruction(),
            tools=tools,
            temperature=0.6,
            max_output_tokens=300,
        )

    def _init_chat(self, conversation_history: list, model: Optional[str] = None):
        """Initializes or restores multi-turn chat session with Gemini."""
        history = []
        for turn in conversation_history:
            try:
                role = "user" if turn.get("role") == "user" else "model"
                content_text = turn.get("content", "")
                if content_text:
                    history.append(types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=content_text)]
                    ))
            except Exception as parse_err:
                logger.warning("Skipping malformed conversation-history turn: %s", parse_err)

        config = self._build_config()
        models_to_try = [model] if model else [self.model_name] + [m for m in self.fallback_models if m != self.model_name]

        last_err = None
        for candidate in models_to_try:
            try:
                self._chat = self.client.chats.create(
                    model=candidate,
                    history=history,
                    config=config
                )
                self.model_name = candidate
                logger.info("Chat initialized with Gemini model: %s", candidate)
                return
            except Exception as err:
                logger.warning("Failed to create chat with model %s: %s", candidate, err)
                last_err = err

        logger.error("All candidate models failed to initialize chat: %s", last_err)
        raise last_err

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
            logger.error("GenAI client is uninitialized (missing API key or client initialization failure).")
            return "I am having trouble connecting to my scheduling system right now. Could you please repeat that?", "error"

        # List of models to try in priority order
        models_to_attempt = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]
        response_text = ""
        last_exception = None

        for candidate in models_to_attempt:
            try:
                if self._chat is None or self.model_name != candidate:
                    self._init_chat(conversation_history, model=candidate)

                logger.info(
                    "Executing chat.send_message with model '%s' input_characters=%s",
                    candidate,
                    len(user_input),
                )
                response = self._chat.send_message(user_input)
                response_text = (response.text or "").strip()
                if response_text:
                    # Successfully received output
                    self.model_name = candidate
                    break
            except Exception as e:
                last_exception = e
                logger.warning(
                    "Gemini API attempt on model '%s' failed: %s (%s). Attempting fallback...",
                    candidate,
                    e,
                    type(e).__name__
                )
                self._chat = None

        if not response_text:
            logger.exception("All Gemini models failed in process_dialog. Last error: %s", last_exception)
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