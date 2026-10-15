import asyncio
import base64
import json
import logging
import re
from datetime import timedelta
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

from django.conf import settings
from django.core import signing
from django.utils import timezone

from channels.db import database_sync_to_async

from core.consumers import AUDIO_CHUNK_BYTES, AUDIO_SAMPLE_WIDTH, VoiceConsumer
from core.models import Appointment, Reminder
from core.reminders.service import ReminderService
from core.services.appointment_service import AppointmentService
from core.repositories import DataRepository
from core.views import CALL_TOKEN_SALT

logger = logging.getLogger(__name__)


ACKNOWLEDGE_UTTERANCES = {"ok", "okay", "got it", "alright", "thanks", "thank you", "understood"}
SNOOZE_UTTERANCES = {"snooze", "snooze it", "remind me later", "later"}
SNOOZE_REQUEST = re.compile(
    r"^(?:(?:please|can you|could you|would you|i want to|i'd like to)\s+)*"
    r"(?:snooze(?: it| reminder| this reminder)?"
    r"(?: for (?P<minutes>\d+) (?:minutes?|mins?))?|"
    r"remind me (?:again )?(?:later|in (?P<remind_minutes>\d+) (?:minutes?|mins?)))"
    r"(?: please)?$"
)
SNOOZE_NEGATION = re.compile(r"\b(?:don't|do not|not|never)\s+(?:please\s+)?snooze\b")
YES_UTTERANCES = {"yes", "yeah", "confirm", "go ahead", "do it", "okay", "ok"}
NO_UTTERANCES = {"no", "cancel that", "never mind"}

DOCTOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "propose_reschedule",
            "description": "Propose a new time for the current appointment; this never changes it until a later explicit confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_start_time": {"type": "string"},
                    "new_end_time": {"type": "string"},
                },
                "required": ["new_start_time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check whether the doctor is free at a requested time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                },
                "required": ["start_time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_today_schedule",
            "description": "List the remaining scheduled appointments today.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def _format_clinic_time(value, timezone_name, date=False):
    local_value = timezone.localtime(value, ZoneInfo(timezone_name))
    return local_value.strftime("%A, %B %d at %I:%M %p" if date else "%I:%M %p")


class DoctorVoiceConsumer(VoiceConsumer):
    async def authorize(self):
        if not settings.DEBUG:
            return False
        query = parse_qs(self.scope.get("query_string", b"").decode("ascii", errors="ignore"))
        token = query.get("token", [None])[0]
        if not token:
            return False
        try:
            payload = signing.TimestampSigner(salt=CALL_TOKEN_SALT).unsign_object(
                token,
                max_age=120,
            )
            reminder_id = int(payload["reminder_id"])
        except (signing.BadSignature, KeyError, TypeError, ValueError):
            return False
        valid = await database_sync_to_async(
            lambda: Reminder.objects.filter(
                pk=reminder_id,
                status__in=Reminder.OPEN_STATUSES,
                appointment__status="Scheduled",
                appointment__start_time__gt=timezone.now(),
            ).exists(),
            thread_sensitive=True,
        )()
        if valid:
            self.reminder_id = reminder_id
        return valid

    def get_session_prefix(self):
        return "doc:"

    async def on_ready(self):
        try:
            context = await database_sync_to_async(
                ReminderService.get_call_context,
                thread_sensitive=True,
            )(self.reminder_id)
            await database_sync_to_async(
                ReminderService.record_call_started,
                thread_sensitive=True,
            )(self.reminder_id)
            greeting = self._build_greeting(context)
            self.conversation_history = [{"role": "assistant", "content": greeting}]
            start_time = timezone.localtime(
                context["start_time"],
                ZoneInfo(context["clinic_tz"]),
            )
            await self.send(text_data=json.dumps({
                "type": "call_ready",
                "reminder_id": context["reminder_id"],
                "appointment_id": context["appointment_id"],
                "patient_name": context["patient_name"],
                "start_time": start_time.isoformat(),
                "minutes_left": context["minutes_left"],
                "snooze_allowed": context["snooze_allowed"],
            }))
            await self._send_doctor_message(greeting, is_welcome=True)
        except Exception:
            logger.exception("Doctor voice call initialization failed reminder_id=%s", self.reminder_id)
            await self.close(code=4404)

    @staticmethod
    def _build_greeting(context):
        when = (
            "starting now" if context["minutes_left"] <= 0
            else "in 1 minute" if context["minutes_left"] == 1
            else f"in {context['minutes_left']} minutes"
        )
        start = _format_clinic_time(context["start_time"], context["clinic_tz"])
        parts = [
            f"{context['doctor_name']}, {context['patient_name']} is at {start}, {when}."
        ]
        if settings.VOICE_INCLUDE_PURPOSE and context["purpose"]:
            parts.append(f"Reason: {context['purpose']}.")
        if context["snooze_allowed"]:
            parts.append(
                f"Say okay, snooze for {settings.SNOOZE_MINUTES} minutes or another number "
                "of minutes, or move it."
            )
        else:
            parts.append("That's too close to snooze. Say okay, or tell me to move it.")
        return " ".join(parts)

    async def _send_doctor_message(self, text, user_text="", tool_status=None, is_welcome=False):
        self.turn_counter += 1
        turn_id = self.turn_counter
        audio_bytes, sample_rate = await asyncio.to_thread(self.tts.synthesize, text)
        if not audio_bytes:
            raise RuntimeError("Piper returned no audio for the doctor voice response")
        if len(audio_bytes) % AUDIO_SAMPLE_WIDTH:
            raise ValueError("Piper doctor response is not aligned to 16-bit samples")

        for sequence, offset in enumerate(range(0, len(audio_bytes), AUDIO_CHUNK_BYTES)):
            await self.send(text_data=json.dumps({
                "type": "audio_chunk",
                "turn_id": turn_id,
                "sequence": sequence,
                "format": "pcm_s16le",
                "channels": 1,
                "sample_rate": sample_rate,
                "audio_payload": base64.b64encode(
                    audio_bytes[offset:offset + AUDIO_CHUNK_BYTES]
                ).decode("ascii"),
            }))
        await self.send(text_data=json.dumps({
            "type": "agent_response",
            "turn_id": turn_id,
            "is_welcome": is_welcome,
            "user_transcription": user_text,
            "caption": text,
            "tool_status": tool_status,
        }))

    async def run_dialog_and_respond(self, user_text):
        turn_id = self.turn_counter + 1
        context = await database_sync_to_async(
            ReminderService.get_call_context,
            thread_sensitive=True,
        )(self.reminder_id)
        response, tool_status, terminal = await asyncio.to_thread(
            self._doctor_turn,
            user_text,
            context,
            turn_id,
        )
        self.conversation_history.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": response},
        ])
        await self._send_doctor_message(
            response,
            user_text=user_text,
            tool_status=tool_status,
        )
        try:
            await database_sync_to_async(
                DataRepository.save_transcript,
                thread_sensitive=True,
            )(
                session_id=self.session_id,
                full_transcript=f"User: {user_text} | Agent: {response}",
            )
        except Exception:
            logger.exception("Doctor voice transcript persistence failed")
        if terminal and settings.CALL_AUTO_HANGUP_AFTER_TERMINAL:
            await self.send(text_data=json.dumps({
                "type": "call_ended",
                "reason": tool_status,
            }))
            await self.close(code=1000)

    def _doctor_turn(self, user_text, context, turn_id):
        text = _normalize(user_text)
        if self._pending_action_expired():
            self.pending_action = None
            if text in YES_UTTERANCES:
                return "That proposal expired. Please ask me to move it again.", "error", False

        if SNOOZE_NEGATION.search(text):
            return "Okay, I won't snooze it. You can say okay to stop reminders or tell me a new appointment time.", None, False
        snooze_match = SNOOZE_REQUEST.fullmatch(text)
        if text in SNOOZE_UTTERANCES or snooze_match:
            self.pending_action = None
            requested_minutes = None
            if snooze_match:
                duration = snooze_match.group("minutes") or snooze_match.group("remind_minutes")
                requested_minutes = int(duration) if duration else None
            return self._snooze_reminder(context, requested_minutes)

        if getattr(self, "pending_action", None):
            if turn_id <= self.pending_action["proposed_in_turn"]:
                return "Please confirm the proposed time in a new turn.", None, False
            if text in YES_UTTERANCES:
                result = AppointmentService().reschedule(
                    context["appointment_id"],
                    self.pending_action["new_start_time"],
                    self.pending_action.get("new_end_time"),
                )
                if not result["ok"]:
                    self.pending_action = None
                    return result["message"], "error", False
                self.pending_action = None
                start_display = _format_clinic_time(
                    result["start_time"],
                    context["clinic_tz"],
                    date=True,
                )
                return (
                    f"Done. {result['patient_name']} is now at {start_display}. "
                    "The reminder has been reset.",
                    "rescheduled",
                    True,
                )
            if text in NO_UTTERANCES:
                self.pending_action = None
                return "Okay, I left the appointment unchanged.", "proposal_discarded", False
            return "Please say yes to confirm that move, or no to leave it unchanged.", None, False

        if text in ACKNOWLEDGE_UTTERANCES:
            result = ReminderService.acknowledge(self.reminder_id)
            return result["message"] if not result["ok"] else "Okay. I won't remind you again.", "reminder_acknowledged", True
        return self._ask_doctor_model(user_text, context, turn_id)

    def _snooze_reminder(self, context, minutes=None):
        result = ReminderService.snooze(self.reminder_id, minutes=minutes)
        if not result["ok"]:
            if result.get("reason") == "invalid_duration":
                return "Please ask for a snooze of at least 1 minute.", "snooze_denied", False
            if result.get("reason") == "too_close_to_start":
                return (
                    f"{context['patient_name']} starts in {result['minutes_left']} minutes, "
                    "too close to snooze. Say okay or tell me to move it.",
                    "snooze_denied",
                    False,
                )
            return "That reminder is no longer available to snooze.", "error", False
        until = _format_clinic_time(result["until"], context["clinic_tz"])
        return (
            f"Snoozed for {result['minutes']} minutes. I'll remind you again at {until}.",
            "reminder_snoozed",
            True,
        )

    def _pending_action_expired(self):
        action = getattr(self, "pending_action", None)
        return bool(
            action
            and timezone.now() - action["created_at"] > timedelta(minutes=2)
        )

    def _ask_doctor_model(self, user_text, context, turn_id):
        if self.llm.client is None:
            return "I can't reach the scheduling assistant right now. You can say okay or snooze.", "error", False

        now_local = timezone.localtime(context["now"], ZoneInfo(context["clinic_tz"]))
        start_local = timezone.localtime(context["start_time"], ZoneInfo(context["clinic_tz"]))
        prompt = (
            f"You are Janus, {context['doctor_name']}'s brief voice assistant. "
            "Keep responses to one or two spoken sentences. Times are in the clinic timezone "
            f"{context['clinic_tz']}. Current time: {now_local.isoformat()}. "
            f"Bound appointment: {context['patient_name']}, starts {start_local.isoformat()}, "
            f"purpose {context['purpose']!r}, reminder {context['reminder_status']}. "
            "Never ask for or invent IDs. For a requested move, call propose_reschedule with an "
            "ISO date-time in the clinic timezone; do not claim it is changed. A snooze request "
            f"without a duration uses {settings.SNOOZE_MINUTES} minutes. "
            "An explicitly requested whole-minute snooze duration is honored when it fits before "
            "the appointment, and snoozing never changes the appointment. "
            "If snooze is too close to the start, offer okay or moving it."
        )
        model_name = settings.DOCTOR_GROQ_MODEL or self.llm.model_name
        try:
            completion = self.llm.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_text},
                ],
                tools=DOCTOR_TOOLS,
                tool_choice="auto",
                max_completion_tokens=220,
                reasoning_effort="low",
            )
        except Exception:
            logger.exception("Doctor voice model request failed")
            return "I couldn't process that request just now. Please try again.", "error", False

        message = completion.choices[0].message
        if not message.tool_calls:
            return message.content or "How would you like to handle this reminder?", None, False
        tool = message.tool_calls[0]
        try:
            arguments = json.loads(tool.function.arguments or "{}")
        except json.JSONDecodeError:
            return "I couldn't understand the requested time. Please say it another way.", "error", False

        appointment_service = AppointmentService()
        if tool.function.name == "propose_reschedule":
            new_start = arguments.get("new_start_time")
            new_end = arguments.get("new_end_time")
            if not new_start:
                return "What date and time should I move it to?", None, False
            old_appointment = Appointment.objects.get(pk=context["appointment_id"])
            try:
                parsed_start = appointment_service._parse_datetime(
                    new_start,
                    context["clinic_tz"],
                )
                parsed_end = (
                    appointment_service._parse_datetime(new_end, context["clinic_tz"])
                    if new_end
                    else parsed_start + (old_appointment.end_time - old_appointment.start_time)
                )
            except (TypeError, ValueError):
                return "I couldn't understand that appointment time. Please say it another way.", "error", False
            availability = appointment_service.check_availability(
                parsed_start,
                parsed_end,
                exclude_appointment_id=context["appointment_id"],
            )
            if not availability["ok"]:
                return availability["message"], "conflict_detected", False
            if availability["start_time"] <= timezone.now():
                return "That time is in the past. Please choose a future time.", "error", False
            if availability["start_time"] == old_appointment.start_time:
                return "That is already the appointment time. Please choose another time.", "error", False
            self.pending_action = {
                "new_start_time": availability["start_time"].isoformat(),
                "new_end_time": availability["end_time"].isoformat() if new_end else None,
                "proposed_in_turn": turn_id,
                "created_at": timezone.now(),
            }
            proposed_time = _format_clinic_time(
                availability["start_time"],
                context["clinic_tz"],
                date=True,
            )
            return (
                f"Move {context['patient_name']} to {proposed_time}. "
                "Shall I do it?",
                "reschedule_proposed",
                False,
            )
        if tool.function.name == "check_availability":
            try:
                result = appointment_service.check_availability(
                    arguments.get("start_time"),
                    arguments.get("end_time"),
                    exclude_appointment_id=context["appointment_id"],
                )
            except (TypeError, ValueError):
                return "I couldn't understand that time. Please say it another way.", "error", False
            if result["ok"]:
                return "That time is available.", "availability_checked", False
            return result["message"], "conflict_detected", False
        if tool.function.name == "get_today_schedule":
            schedule = appointment_service.get_today_schedule()
            if not schedule:
                return "There are no more scheduled appointments today.", "schedule_checked", False
            lines = [
                f"{item['patient_name']} at {_format_clinic_time(item['start_time'], context['clinic_tz'])}"
                for item in schedule
            ]
            return "Today's remaining appointments are " + ", ".join(lines) + ".", "schedule_checked", False
        return "I can't perform that action.", "error", False
