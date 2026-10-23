import asyncio
import base64
import json
from datetime import timedelta, timezone as datetime_timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from channels.db import database_sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.core.management import call_command
from django.core import signing
from django.test import TransactionTestCase, override_settings
from django.utils import timezone as django_timezone

from .models import Appointment, Doctor, Reminder, ReminderEvent, Task, Transcript
from .consumers import VoiceConsumer
from .doc_consumers import DoctorAssistantConsumer, DoctorVoiceConsumer
from .routing import websocket_urlpatterns
from .reminders.policy import first_fire_at, snooze_decision
from .reminders.service import ReminderService
from .reminders.sweep import claim_and_fire
from .services.appointment_service import AppointmentService
from .services.calendar_service import GoogleCalendarService
from .services.vad_service import VADService


class FakeVADService:
	def __init__(self):
		self.call_count = 0
		self.audio_lengths = []

	def speech_probabilities(self, audio_chunk):
		self.call_count += 1
		self.audio_lengths.append(len(audio_chunk))
		audio = np.frombuffer(audio_chunk, dtype=np.int16)
		return [
			0.95 if np.count_nonzero(audio[start:start + 512]) / 512 > 0.5 else 0.05
			for start in range(0, len(audio) - 511, 512)
		]

	def is_speech(self, audio_chunk, sampling_rate=16000, threshold=0.5):
		return any(
			probability >= threshold
			for probability in self.speech_probabilities(audio_chunk)
		)


class FakeSTTService:
	last_audio_parameters = None

	def __init__(self, model_size, language):
		pass

	def transcribe(self, audio_data):
		type(self).last_audio_parameters = (audio_data.dtype, len(audio_data))
		return "test transcription"


class FakeLLMService:
	last_tool_status = None

	def stream_dialog(self, conversation_history, user_input):
		yield "Dr. Melvin is available. "
		yield "Would you like to book?"


class FakeTTSService:
	def __init__(self):
		self.inputs = []

	def synthesize(self, text):
		self.inputs.append(text)
		return b"test audio", 22050


class FakeGroqStream:
	def __init__(self, chunks):
		self.chunks = chunks

	def __iter__(self):
		return iter(self.chunks)

	def close(self):
		pass


def groq_content_chunk(content):
	return MagicMock(choices=[MagicMock(delta=MagicMock(content=content, tool_calls=None))])


def groq_tool_call_chunk(call_id, name, arguments, index=0):
	tool_call = MagicMock(
		index=index,
		id=call_id,
		function=SimpleNamespace(name=name, arguments=arguments),
	)
	return MagicMock(choices=[MagicMock(
		delta=MagicMock(content=None, tool_calls=[tool_call])
	)])


class VoiceConsumerTests(TransactionTestCase):
	def setUp(self):
		self.fake_vad = FakeVADService()
		self.fake_tts = FakeTTSService()
		self.patches = [
			patch("core.consumers.VADService", return_value=self.fake_vad),
			patch("core.consumers.get_vad_service", return_value=self.fake_vad),
			patch("core.consumers.STTService", return_value=FakeSTTService("base.en", "en")),
			patch("core.consumers.LLMService", return_value=FakeLLMService()),
			patch("core.consumers.TTSService", return_value=self.fake_tts),
		]
		for service_patch in self.patches:
			service_patch.start()
			self.addCleanup(service_patch.stop)

	async def connect_communicator(self):
		self.communicator = WebsocketCommunicator(
			URLRouter(websocket_urlpatterns),
			"/ws/voice/",
		)
		connected, _ = await self.communicator.connect()
		self.assertTrue(connected)
		welcome = await self.communicator.receive_json_from(timeout=3)
		self.assertEqual(welcome["type"], "welcome")
		self.assertEqual(welcome["caption"], (
			"Hello, I’m Janus, Dr. Melvin’s appointment scheduler. "
			"I can help you check availability or schedule a visit. How may I help you today?"
		))
		welcome_audio = []
		while True:
			message = await self.communicator.receive_json_from(timeout=3)
			if message["type"] == "agent_response":
				self.assertTrue(message["is_welcome"])
				self.assertEqual(message["turn_id"], 1)
				break
			welcome_audio.append(message)
		self.assertTrue(welcome_audio)
		self.assertTrue(all(message["type"] == "audio_chunk" for message in welcome_audio))
		self.assertEqual(
			b"".join(base64.b64decode(message["audio_payload"]) for message in welcome_audio),
			b"test audio",
		)
		await self.communicator.send_to(text_data=json.dumps({
			"action": "playback_complete",
			"turn_id": 1,
		}))

	def test_vad_service_preserves_probability_for_each_window(self):
		probabilities = iter((0.2, 0.8, 0.6))
		service = VADService.__new__(VADService)
		service.model = MagicMock(
			side_effect=lambda *_: SimpleNamespace(item=lambda: next(probabilities))
		)

		result = service.speech_probabilities(b"\x00\x00" * (512 * 3))

		self.assertEqual(result, [0.2, 0.8, 0.6])
		self.assertEqual(service.model.call_count, 3)

	async def test_text_turn_returns_audio_and_saves_transcript(self):
		await self.connect_communicator()
		try:
			with self.assertLogs("core", level="INFO") as pipeline_logs:
				await self.communicator.send_to(text_data=json.dumps({
					"action": "process_text_turn",
					"text": "hello",
				}))

				processing = await self.communicator.receive_json_from(timeout=3)
				stream_messages = []
				while True:
					message = await self.communicator.receive_json_from(timeout=3)
					if message["type"] == "agent_response":
						response = message
						break
					stream_messages.append(message)

			self.assertEqual(processing["type"], "processing")
			log_output = "\n".join(pipeline_logs.output)
			for stage in (
				"Pipeline receive: action=process_text_turn",
				"Pipeline dialog: started",
				"Pipeline dialog: completed",
				"Pipeline TTS: streaming synthesis started",
				"Pipeline TTS: sentence synthesized",
				"Pipeline persistence: transcript saved",
				"Pipeline delivery: response sent",
				"Pipeline turn: total",
			):
				self.assertIn(stage, log_output)
			self.assertRegex(log_output, r"Pipeline turn: total .*elapsed_ms=\d+(?:\.\d+)?")
			self.assertRegex(log_output, r"Pipeline persistence: transcript saved .*elapsed_ms=\d+(?:\.\d+)?")
			self.assertRegex(log_output, r"Pipeline delivery: response sent .*elapsed_ms=\d+(?:\.\d+)?")
			self.assertNotIn("hello", log_output)

			self.assertEqual(response["type"], "agent_response")
			self.assertEqual(response["caption"], "Dr. Melvin is available. Would you like to book?")
			self.assertEqual(response["turn_id"], 2)
			self.assertEqual(
				self.fake_tts.inputs,
				[
					"Hello, I’m Janus, Dr. Melvin’s appointment scheduler. "
					"I can help you check availability or schedule a visit. How may I help you today?",
					"Dr. Melvin is available.",
					"Would you like to book?",
				],
			)
			self.assertTrue(stream_messages)
			self.assertTrue(all(message["type"] == "audio_chunk" for message in stream_messages))
			self.assertEqual(
				b"".join(base64.b64decode(message["audio_payload"]) for message in stream_messages),
				b"test audio" * (len(self.fake_tts.inputs) - 1),
			)
			transcript = await database_sync_to_async(
				lambda: Transcript.objects.get(session_id__startswith="session_")
			)()
			self.assertEqual(
				transcript.full_transcript,
				"User: hello | Agent: Dr. Melvin is available. Would you like to book?",
			)
		finally:
			await self.communicator.disconnect()

	async def test_patient_negative_followup_plays_goodbye_before_call_ended(self):
		await self.connect_communicator()
		try:
			with patch.object(
				FakeLLMService,
				"stream_dialog",
				return_value=iter(["Is there anything else I can help you with?"]),
			):
				await self.communicator.send_to(text_data=json.dumps({
					"action": "process_text_turn",
					"text": "check my appointment",
				}))
				await self.communicator.receive_json_from(timeout=3)
				while True:
					message = await self.communicator.receive_json_from(timeout=3)
					if message["type"] == "agent_response":
						self.assertIn("anything else", message["caption"].lower())
						self.assertFalse(message["terminal"])
						break

			await self.communicator.send_to(text_data=json.dumps({
				"action": "process_text_turn",
				"text": "no, that's all",
			}))
			await self.communicator.receive_json_from(timeout=3)
			audio_chunks = []
			while True:
				message = await self.communicator.receive_json_from(timeout=3)
				if message["type"] == "agent_response":
					self.assertEqual(message["caption"], "Thank you. Goodbye.")
					self.assertTrue(message["terminal"])
					break
				audio_chunks.append(message)

			self.assertTrue(audio_chunks)
			self.assertTrue(all(message["type"] == "audio_chunk" for message in audio_chunks))
			self.assertEqual(
				b"".join(base64.b64decode(message["audio_payload"]) for message in audio_chunks),
				b"test audio",
			)
			ended = await self.communicator.receive_json_from(timeout=3)
			self.assertEqual(
				ended,
				{"type": "call_ended", "reason": "conversation_complete"},
			)

			await self.communicator.send_to(text_data=json.dumps({"action": "end_call"}))
		finally:
			await self.communicator.disconnect()

	async def test_markdown_asterisks_are_removed_from_spoken_and_displayed_response(self):
		await self.connect_communicator()
		try:
			with patch.object(
				FakeLLMService,
				"stream_dialog",
				return_value=iter(["**Dr. Melvin** is available."]),
			):
				await self.communicator.send_to(text_data=json.dumps({
					"action": "process_text_turn",
					"text": "hello",
				}))

				await self.communicator.receive_json_from(timeout=3)
				while True:
					message = await self.communicator.receive_json_from(timeout=3)
					if message["type"] == "agent_response":
						break

			self.assertEqual(message["caption"], "Dr. Melvin is available.")
			self.assertEqual(self.fake_tts.inputs[-1], "Dr. Melvin is available.")
			transcript = await database_sync_to_async(
				lambda: Transcript.objects.get(session_id__startswith="session_")
			)()
			self.assertEqual(
				transcript.full_transcript,
				"User: hello | Agent: Dr. Melvin is available.",
			)
		finally:
			await self.communicator.disconnect()

	async def test_punctuation_only_transcription_returns_retry_prompt(self):
		await self.connect_communicator()
		try:
			with patch.object(FakeSTTService, "transcribe", return_value="."):
				await self.communicator.send_to(bytes_data=b"\x01\x00" * 8000)
				await self.communicator.send_to(bytes_data=b"\x00\x00" * 11712)
				await self.communicator.send_to(bytes_data=b"\x00\x00" * 512)

				processing = await self.communicator.receive_json_from(timeout=3)
				while True:
					message = await self.communicator.receive_json_from(timeout=3)
					if message["type"] == "agent_response":
						break

			self.assertEqual(processing["type"], "processing")
			self.assertEqual(message["user_transcription"], "")
			self.assertEqual(
				message["caption"],
				"I didn’t catch that. Could you please repeat?",
			)
			self.assertEqual(
				self.fake_tts.inputs[-1],
				"I didn’t catch that. Could you please repeat?",
			)
		finally:
			await self.communicator.disconnect()

	@override_settings(DEBUG=True)
	async def test_doctor_reminder_call_snoozes_without_changing_appointment(self):
		def create_reminder():
			doctor, _ = Doctor.objects.get_or_create(
				pk=Doctor.SINGLETON_ID,
				defaults={
					"name": "Dr. Melvin",
					"email": "doctor@example.test",
					"timezone": "Asia/Karachi",
				},
			)
			start_time = django_timezone.now() + timedelta(hours=1)
			appointment = Appointment.objects.create(
				doctor=doctor,
				caller_name="Patient Example",
				start_time=start_time,
				end_time=start_time + timedelta(minutes=30),
				purpose="Routine checkup",
			)
			return ReminderService.create_for_appointment(appointment).pk, start_time

		reminder_id, start_time = await database_sync_to_async(create_reminder)()
		token = signing.TimestampSigner(salt="core.reminders.call").sign_object(
			{"reminder_id": reminder_id}
		)
		communicator = WebsocketCommunicator(
			URLRouter(websocket_urlpatterns),
			f"/ws/doc-voice/?token={token}",
		)
		connected, denial = await communicator.connect()
		self.assertTrue(connected, denial)
		try:
			ready = await communicator.receive_json_from(timeout=3)
			self.assertEqual(ready["type"], "call_ready")
			self.assertEqual(ready["patient_name"], "Patient Example")
			while True:
				message = await communicator.receive_json_from(timeout=3)
				if message["type"] == "agent_response":
					self.assertTrue(message["is_welcome"])
					self.assertNotIn("Dr. Dr.", message["caption"])
					break

			with patch.object(FakeSTTService, "transcribe", return_value="snooze"), patch.object(
				self.fake_vad,
				"speech_probabilities",
				side_effect=[[0.95], [0.05], [0.95], [0.95], [0.05] * 22],
			):
				for is_speech in (True, False, True, True):
					value = b"\x01\x00" if is_speech else b"\x00\x00"
					await communicator.send_to(bytes_data=value * 512)
				await communicator.send_to(bytes_data=b"\x00\x00" * 11264)

				processing = await communicator.receive_json_from(timeout=3)
				self.assertEqual(processing["type"], "processing")
				while True:
					message = await communicator.receive_json_from(timeout=3)
					if message["type"] == "agent_response":
						self.assertEqual(message["user_transcription"], "snooze")
						self.assertEqual(message["tool_status"], "reminder_snoozed")
						break
			ended = await communicator.receive_json_from(timeout=3)
			self.assertEqual(ended, {"type": "call_ended", "reason": "reminder_snoozed"})
			reminder = await database_sync_to_async(Reminder.objects.select_related("appointment").get)(
				pk=reminder_id,
			)
			self.assertEqual(reminder.status, Reminder.Status.SNOOZED)
			self.assertLess(
				abs((reminder.next_fire_at - reminder.last_call_at - timedelta(minutes=15)).total_seconds()),
				2,
			)
			self.assertEqual(reminder.appointment.start_time, start_time)
		finally:
			await communicator.disconnect()

	@override_settings(DEBUG=True)
	async def test_general_doctor_assistant_websocket_connects_without_a_reminder(self):
		await database_sync_to_async(
			lambda: Doctor.objects.get_or_create(
				pk=Doctor.SINGLETON_ID,
				defaults={
					"name": "Dr. Melvin",
					"email": "doctor@example.test",
					"timezone": "Asia/Karachi",
				},
			)
		)()
		token = signing.TimestampSigner(salt="core.reminders.call").sign_object(
			{"doctor_assistant": True}
		)
		communicator = WebsocketCommunicator(
			URLRouter(websocket_urlpatterns),
			f"/ws/doc-assistant/?token={token}",
		)
		connected, denial = await communicator.connect()
		self.assertTrue(connected, denial)
		try:
			ready = await communicator.receive_json_from(timeout=3)
			self.assertTrue(ready["doctor_assistant"])
			self.assertEqual(ready["doctor_name"], "Dr. Melvin")
			while True:
				message = await communicator.receive_json_from(timeout=3)
				if message["type"] == "agent_response":
					self.assertTrue(message["is_welcome"])
					self.assertIn("upcoming appointments", message["caption"])
					break
		finally:
			await communicator.disconnect()

	async def test_binary_pcm_is_converted_to_float32_for_transcription(self):
		await self.connect_communicator()
		try:
			# Speech chunk (16,000 bytes = 0.5s speech)
			speech_chunk = b"\x01\x00" * 8000
			# Five hundred milliseconds must not endpoint the utterance.
			silence_chunk = b"\x00\x00" * 8000
			await self.communicator.send_to(bytes_data=speech_chunk)
			await self.communicator.send_to(bytes_data=silence_chunk)
			self.assertTrue(await self.communicator.receive_nothing(timeout=0.05))
			await self.communicator.send_to(bytes_data=b"\x00\x00" * 3200)
			await self.communicator.send_to(bytes_data=b"\x00\x00" * 512)

			processing = await self.communicator.receive_json_from(timeout=3)
			stream_messages = []
			while True:
				message = await self.communicator.receive_json_from(timeout=3)
				if message["type"] == "agent_response":
					response = message
					break
				stream_messages.append(message)

			self.assertEqual(processing["type"], "processing")
			self.assertEqual(response["user_transcription"], "test transcription")
			self.assertTrue(stream_messages)
			self.assertEqual(FakeSTTService.last_audio_parameters, (np.dtype(np.float32), 19712))
		finally:
			await self.communicator.disconnect()

	async def test_single_vad_positive_does_not_start_transcription(self):
		await self.connect_communicator()
		try:
			await self.communicator.send_to(bytes_data=b"\x01\x00" * 512)
			await self.communicator.send_to(bytes_data=b"\x00\x00" * 11200)

			self.assertTrue(await self.communicator.receive_nothing(timeout=0.1))
			self.assertEqual(self.fake_vad.call_count, 2)
		finally:
			await self.communicator.disconnect()

	async def test_speech_starts_when_three_vad_positives_have_a_brief_gap(self):
		await self.connect_communicator()
		try:
			with patch.object(
				self.fake_vad,
				"speech_probabilities",
				side_effect=[[0.95], [0.05], [0.95], [0.95], [0.05] * 22],
			):
				for is_speech in (True, False, True, True):
					value = b"\x01\x00" if is_speech else b"\x00\x00"
					await self.communicator.send_to(bytes_data=value * 512)

				self.assertTrue(await self.communicator.receive_nothing(timeout=0.05))
				await self.communicator.send_to(bytes_data=b"\x00\x00" * 11264)

				processing = await self.communicator.receive_json_from(timeout=3)
				while True:
					message = await self.communicator.receive_json_from(timeout=3)
					if message["type"] == "agent_response":
						break

			self.assertEqual(processing["type"], "processing")
			self.assertEqual(message["user_transcription"], "test transcription")
		finally:
			await self.communicator.disconnect()

	async def test_vad_preserves_partial_windows_between_audio_frames(self):
		await self.connect_communicator()
		try:
			await self.communicator.send_to(bytes_data=b"\x01\x00" * 256)
			await self.communicator.send_to(text_data="not-json")
			self.assertEqual(
				(await self.communicator.receive_json_from(timeout=3))["type"],
				"error",
			)
			self.assertEqual(self.fake_vad.call_count, 0)

			await self.communicator.send_to(bytes_data=b"\x01\x00" * 256)
			await self.communicator.send_to(text_data="not-json")
			self.assertEqual(
				(await self.communicator.receive_json_from(timeout=3))["type"],
				"error",
			)
			self.assertEqual(self.fake_vad.call_count, 1)
			self.assertEqual(self.fake_vad.audio_lengths, [1024])

			await self.communicator.send_to(bytes_data=b"\x01\x00" * 1536)
			await self.communicator.send_to(bytes_data=b"\x00\x00" * 8000)
			self.assertTrue(await self.communicator.receive_nothing(timeout=0.05))
			await self.communicator.send_to(bytes_data=b"\x00\x00" * 3200)
			await self.communicator.send_to(bytes_data=b"\x00\x00" * 64)
			processing = await self.communicator.receive_json_from(timeout=3)
			while True:
				message = await self.communicator.receive_json_from(timeout=3)
				if message["type"] == "agent_response":
					break

			self.assertEqual(processing["type"], "processing")
			self.assertEqual(self.fake_vad.call_count, 5)
			self.assertTrue(all(length % 1024 == 0 for length in self.fake_vad.audio_lengths))
			self.assertEqual(FakeSTTService.last_audio_parameters, (np.dtype(np.float32), 13312))
		finally:
			await self.communicator.disconnect()

	async def test_empty_transcription_returns_retry_prompt_and_finishes_turn(self):
		await self.connect_communicator()
		try:
			with patch.object(FakeSTTService, "transcribe", return_value=""):
				await self.communicator.send_to(bytes_data=b"\x01\x00" * 8000)
				await self.communicator.send_to(bytes_data=b"\x00\x00" * 11712)
				await self.communicator.send_to(bytes_data=b"\x00\x00" * 512)

				processing = await self.communicator.receive_json_from(timeout=3)
				audio_message = await self.communicator.receive_json_from(timeout=3)
				response = await self.communicator.receive_json_from(timeout=3)

			self.assertEqual(processing["type"], "processing")
			self.assertEqual(audio_message["type"], "audio_chunk")
			self.assertEqual(response["type"], "agent_response")
			self.assertEqual(response["user_transcription"], "")
			self.assertEqual(
				response["caption"],
				"I didn’t catch that. Could you please repeat?",
			)
		finally:
			await self.communicator.disconnect()

	async def test_audio_does_not_cancel_turn_while_assistant_is_processing(self):
		consumer = VoiceConsumer()
		consumer.active_task = asyncio.create_task(asyncio.sleep(30))
		consumer.response_audio_active = False
		consumer.vad = self.fake_vad
		try:
			await consumer.receive(bytes_data=b"\x01\x00" * 512)

			self.assertFalse(consumer.active_task.done())
			self.assertEqual(self.fake_vad.call_count, 0)
		finally:
			consumer.active_task.cancel()
			try:
				await consumer.active_task
			except asyncio.CancelledError:
				pass

	async def test_speech_during_assistant_playback_cancels_reply_automatically(self):
		consumer = VoiceConsumer()
		consumer.session_id = "test-session"
		consumer.active_task = None
		consumer.audio_buffer = bytearray()
		consumer.pre_speech_buffer = bytearray()
		consumer.speech_candidate_buffer = bytearray()
		consumer.speech_candidate_positive_bytes = 0
		consumer.speech_candidate_confidence_window = []
		consumer.speech_candidate_silence_bytes = 0
		consumer.speech_detected = False
		consumer.consecutive_silence_bytes = 0
		consumer.speech_started_at = None
		consumer.speech_vad_elapsed_seconds = 0.0
		consumer.vad_remainder = bytearray()
		consumer.vad = self.fake_vad
		consumer.send = AsyncMock()
		consumer.turn_counter = 4
		consumer.response_audio_active = True
		consumer.response_audio_turn_id = 4
		response_task = asyncio.create_task(asyncio.sleep(30))
		consumer.active_task = response_task

		try:
			for _ in range(6):
				await consumer.receive(bytes_data=b"\x01\x00" * 512)

			self.assertIsNone(consumer.active_task)
			self.assertFalse(consumer.response_audio_active)
			self.assertIsNone(consumer.response_audio_turn_id)
			consumer.send.assert_awaited_once()
			stopped_message = json.loads(consumer.send.await_args.kwargs["text_data"])
			self.assertEqual(stopped_message, {"status": "audio_stopped", "turn_id": 4})
			self.assertTrue(consumer.speech_detected)
		finally:
			response_task.cancel()
			try:
				await response_task
			except asyncio.CancelledError:
				pass

	async def test_click_and_short_cough_do_not_interrupt_but_sustained_speech_does(self):
		consumer = VoiceConsumer()
		consumer.session_id = "test-session"
		consumer.active_task = None
		consumer.audio_buffer = bytearray()
		consumer.pre_speech_buffer = bytearray()
		consumer.speech_candidate_buffer = bytearray()
		consumer.speech_candidate_positive_bytes = 0
		consumer.speech_candidate_confidence_window = []
		consumer.speech_candidate_silence_bytes = 0
		consumer.speech_detected = False
		consumer.consecutive_silence_bytes = 0
		consumer.speech_started_at = None
		consumer.speech_vad_elapsed_seconds = 0.0
		consumer.vad_remainder = bytearray()
		consumer.vad = self.fake_vad
		consumer.send = AsyncMock()
		consumer.turn_counter = 4
		consumer.response_audio_active = True
		consumer.response_audio_turn_id = 4

		speech_frame = b"\x01\x00" * 512
		silence_frame = b"\x00\x00" * 512
		await consumer.receive(bytes_data=speech_frame)
		for _ in range(3):
			await consumer.receive(bytes_data=silence_frame)
		for _ in range(4):
			await consumer.receive(bytes_data=speech_frame)
		for _ in range(3):
			await consumer.receive(bytes_data=silence_frame)

		self.assertTrue(consumer.response_audio_active)
		consumer.send.assert_not_awaited()
		self.assertFalse(consumer.speech_detected)

		for _ in range(6):
			await consumer.receive(bytes_data=speech_frame)

		self.assertFalse(consumer.response_audio_active)
		self.assertTrue(consumer.speech_detected)
		consumer.send.assert_awaited_once()

	async def test_interrupted_speech_completes_the_regular_request_response_cycle(self):
		await self.connect_communicator()
		try:
			await self.communicator.send_to(text_data=json.dumps({
				"action": "process_text_turn",
				"text": "hello",
			}))
			await self.communicator.receive_json_from(timeout=3)
			while True:
				message = await self.communicator.receive_json_from(timeout=3)
				if message["type"] == "agent_response":
					break

			speech_frame = b"\x01\x00" * 512
			for _ in range(8):
				await self.communicator.send_to(bytes_data=speech_frame)

			stopped = await self.communicator.receive_json_from(timeout=3)
			self.assertEqual(stopped["status"], "audio_stopped")

			await self.communicator.send_to(bytes_data=b"\x00\x00" * 11264)
			processing = await self.communicator.receive_json_from(timeout=3)
			self.assertEqual(processing["type"], "processing")
			while True:
				message = await self.communicator.receive_json_from(timeout=3)
				if message["type"] == "agent_response":
					break

			self.assertEqual(message["user_transcription"], "test transcription")
			self.assertEqual(message["caption"], "Dr. Melvin is available. Would you like to book?")
		finally:
			await self.communicator.disconnect()

	async def test_playback_complete_releases_backend_vad_gate(self):
		consumer = VoiceConsumer()
		consumer.session_id = "test-session"
		consumer.response_audio_active = True
		consumer.response_audio_turn_id = 4

		await consumer.receive(text_data=json.dumps({
			"action": "playback_complete",
			"turn_id": 4,
		}))

		self.assertFalse(consumer.response_audio_active)
		self.assertIsNone(consumer.response_audio_turn_id)



from unittest.mock import MagicMock
from core.services.calendar_service import GoogleCalendarService, CalendarAuthError


class DoctorDashboardTests(TransactionTestCase):
	def setUp(self):
		self.doctor, _ = Doctor.objects.get_or_create(
			pk=Doctor.SINGLETON_ID,
			defaults={
				"name": "Dr. Melvin",
				"email": "doctor@example.test",
				"timezone": "Asia/Karachi",
			},
		)
		self.now = django_timezone.now()

	def create_appointment(self, **overrides):
		values = {
			"doctor": self.doctor,
			"caller_name": "Patient Example",
			"start_time": self.now + timedelta(minutes=10),
			"end_time": self.now + timedelta(minutes=40),
			"purpose": "Routine checkup",
			"status": "Scheduled",
		}
		values.update(overrides)
		return Appointment.objects.create(**values)

	@override_settings(DEBUG=True)
	def test_doctor_dashboard_returns_upcoming_all_and_reminder_data(self):
		due_soon = self.create_appointment()
		self.create_appointment(
			caller_name="Later Patient",
			start_time=self.now + timedelta(hours=2),
			end_time=self.now + timedelta(hours=2, minutes=30),
		)
		self.create_appointment(
			caller_name="Cancelled Patient",
			start_time=self.now + timedelta(minutes=5),
			end_time=self.now + timedelta(minutes=35),
			status="Cancelled",
		)

		response = self.client.get("/api/doc/dashboard/")

		self.assertEqual(response.status_code, 200)
		data = response.json()
		self.assertEqual(len(data["upcoming"]), 2)
		self.assertEqual(len(data["all_appointments"]), 3)
		self.assertEqual([item["id"] for item in data["reminders"]], [due_soon.pk])
		self.assertEqual(data["reminders"][0]["purpose"], "Routine checkup")
		self.assertEqual(data["doctor"]["timezone"], self.doctor.timezone)

	@override_settings(DEBUG=False)
	def test_doctor_dashboard_is_not_available_outside_development(self):
		response = self.client.get("/api/doc/dashboard/")

		self.assertEqual(response.status_code, 404)

	@override_settings(
		REMINDER_LEAD_MINUTES=30,
		REMINDER_MIN_LEAD_TO_REMIND_MINUTES=2,
	)
	def test_reminder_policy_uses_thirty_minute_lead_and_skips_imminent_visits(self):
		now = django_timezone.now()
		start = now + timedelta(minutes=90)
		self.assertEqual(first_fire_at(start, now), start - timedelta(minutes=30))
		self.assertIsNone(first_fire_at(now + timedelta(minutes=1), now))

	@override_settings(SNOOZE_MINUTES=15, SNOOZE_MIN_BUFFER_MINUTES=1)
	def test_snooze_policy_allows_exact_buffer_and_rejects_later(self):
		now = django_timezone.now()
		self.assertTrue(snooze_decision(now, now + timedelta(minutes=16)).allowed)
		self.assertFalse(snooze_decision(now, now + timedelta(minutes=15)).allowed)
		self.assertTrue(
			snooze_decision(now, now + timedelta(minutes=11), minutes=10).allowed
		)
		self.assertFalse(
			snooze_decision(now, now + timedelta(minutes=10), minutes=10).allowed
		)

	@override_settings(SNOOZE_MINUTES=15, SNOOZE_MIN_BUFFER_MINUTES=1)
	def test_snoozing_sets_next_reminder_fifteen_minutes_from_action_only(self):
		action_time = self.now + timedelta(minutes=3)
		appointment = self.create_appointment(
			start_time=self.now + timedelta(minutes=60),
			end_time=self.now + timedelta(minutes=90),
		)
		original_start = appointment.start_time
		reminder = ReminderService.create_for_appointment(appointment, now=self.now)

		result = ReminderService.snooze(reminder.pk, now=action_time)

		self.assertTrue(result["ok"])
		self.assertEqual(result["until"], action_time + timedelta(minutes=15))
		appointment.refresh_from_db()
		reminder.refresh_from_db()
		self.assertEqual(appointment.start_time, original_start)
		self.assertEqual(reminder.next_fire_at, action_time + timedelta(minutes=15))

	@override_settings(SNOOZE_MINUTES=15, SNOOZE_MIN_BUFFER_MINUTES=1)
	def test_custom_snooze_duration_is_used_without_changing_appointment(self):
		action_time = self.now + timedelta(minutes=3)
		appointment = self.create_appointment(
			start_time=self.now + timedelta(minutes=60),
			end_time=self.now + timedelta(minutes=90),
		)
		original_start = appointment.start_time
		reminder = ReminderService.create_for_appointment(appointment, now=self.now)

		result = ReminderService.snooze(reminder.pk, now=action_time, minutes=10)

		self.assertTrue(result["ok"])
		self.assertEqual(result["minutes"], 10)
		self.assertEqual(result["until"], action_time + timedelta(minutes=10))
		appointment.refresh_from_db()
		reminder.refresh_from_db()
		self.assertEqual(appointment.start_time, original_start)
		self.assertEqual(reminder.next_fire_at, action_time + timedelta(minutes=10))

	def test_reminder_move_request_asks_for_date_and_time_without_snoozing(self):
		consumer = object.__new__(DoctorVoiceConsumer)
		consumer.reminder_id = 1
		consumer.pending_action = None
		consumer.reschedule_details = None
		consumer._ask_doctor_model = MagicMock()

		response, status, terminal = consumer._doctor_turn("move it", {}, 1)

		self.assertEqual(response, "What date and time should I move the appointment to?")
		self.assertEqual(status, "reschedule_details_needed")
		self.assertFalse(terminal)
		self.assertEqual(consumer.reschedule_details, "move it")
		consumer._ask_doctor_model.assert_not_called()

	def test_reminder_move_requires_a_specific_date_and_time(self):
		consumer = object.__new__(DoctorVoiceConsumer)
		consumer.reminder_id = 1
		consumer.pending_action = None
		consumer.reschedule_details = None
		consumer._ask_doctor_model = MagicMock()

		response, status, terminal = consumer._doctor_turn(
			"move it to next week at 3 PM",
			{},
			1,
		)

		self.assertEqual(response, "What date and time should I move the appointment to?")
		self.assertEqual(status, "reschedule_details_needed")
		self.assertFalse(terminal)
		consumer._ask_doctor_model.assert_not_called()

	def test_reminder_move_collects_date_and_time_before_proposing_reschedule(self):
		new_start = self.now + timedelta(days=1)
		consumer = object.__new__(DoctorVoiceConsumer)
		consumer.reminder_id = 1
		consumer.pending_action = None
		consumer.reschedule_details = None

		def propose_reschedule(*args):
			consumer.pending_action = {
				"new_start_time": new_start.isoformat(),
				"new_end_time": None,
				"proposed_in_turn": 3,
				"created_at": self.now,
			}
			return "Move it to Tuesday at 3 PM.", "reschedule_proposed", False

		first_response, first_status, _ = consumer._doctor_turn("move it", {}, 1)
		date_response, date_status, _ = consumer._doctor_turn("tomorrow", {}, 2)
		consumer._ask_doctor_model = MagicMock(side_effect=propose_reschedule)
		with patch(
			"core.doc_consumers.AppointmentService.reschedule",
			return_value={
				"ok": True,
				"patient_name": "Patient Example",
				"start_time": new_start,
			},
		) as reschedule:
			response, status, terminal = consumer._doctor_turn(
				"3 PM",
				{"appointment_id": 1, "clinic_tz": self.doctor.timezone},
				3,
			)

		self.assertEqual(first_status, "reschedule_details_needed")
		self.assertEqual(date_status, "reschedule_details_needed")
		self.assertIn("What time", date_response)
		self.assertEqual(status, "rescheduled")
		self.assertTrue(terminal)
		self.assertIn("now scheduled", response)
		consumer._ask_doctor_model.assert_called_once()
		self.assertIn("tomorrow 3 PM", consumer._ask_doctor_model.call_args.args[0])
		self.assertIsNone(consumer.reschedule_details)
		self.assertIsNone(consumer.pending_action)
		reschedule.assert_called_once_with(1, new_start.isoformat(), None)

	def test_reminder_acknowledged_stops_reminder(self):
		consumer = object.__new__(DoctorVoiceConsumer)
		consumer.reminder_id = 17
		consumer.pending_action = None
		consumer.reschedule_details = None
		with patch(
			"core.doc_consumers.ReminderService.acknowledge",
			return_value={"ok": True},
		) as acknowledge:
			response, status, terminal = consumer._doctor_turn("acknowledged", {}, 1)

		self.assertEqual(response, "Okay. I won't remind you again.")
		self.assertEqual(status, "reminder_acknowledged")
		self.assertTrue(terminal)
		acknowledge.assert_called_once_with(17)

	def test_ok_acknowledges_instead_of_confirming_pending_reschedule(self):
		consumer = object.__new__(DoctorVoiceConsumer)
		consumer.reminder_id = 17
		consumer.pending_action = {
			"new_start_time": (self.now + timedelta(days=1)).isoformat(),
			"proposed_in_turn": 1,
			"created_at": self.now,
		}
		consumer.reschedule_details = None
		with patch(
			"core.doc_consumers.ReminderService.acknowledge",
			return_value={"ok": True},
		) as acknowledge, patch(
			"core.doc_consumers.AppointmentService.reschedule",
		) as reschedule:
			response, status, terminal = consumer._doctor_turn("OK", {}, 2)

		self.assertEqual(response, "Okay. I won't remind you again.")
		self.assertEqual(status, "reminder_acknowledged")
		self.assertTrue(terminal)
		self.assertIsNone(consumer.pending_action)
		acknowledge.assert_called_once_with(17)
		reschedule.assert_not_called()

	def test_confirmed_reminder_reschedule_ends_the_call(self):
		new_start = self.now + timedelta(days=1)
		consumer = object.__new__(DoctorVoiceConsumer)
		consumer.reminder_id = 17
		consumer.pending_action = {
			"new_start_time": new_start.isoformat(),
			"new_end_time": None,
			"proposed_in_turn": 1,
			"created_at": self.now,
		}
		consumer.reschedule_details = None
		with patch(
			"core.doc_consumers.AppointmentService.reschedule",
			return_value={
				"ok": True,
				"patient_name": "Patient Example",
				"start_time": new_start,
			},
		) as reschedule:
			response, status, terminal = consumer._doctor_turn("yes", {
				"appointment_id": 17,
				"clinic_tz": self.doctor.timezone,
			}, 2)

		self.assertIn("now scheduled for", response)
		self.assertEqual(status, "rescheduled")
		self.assertTrue(terminal)
		self.assertIsNone(consumer.pending_action)
		reschedule.assert_called_once_with(17, new_start.isoformat(), None)

	def test_reminder_snooze_duration_is_passed_to_time_left_check(self):
		consumer = object.__new__(DoctorVoiceConsumer)
		consumer.reminder_id = 17
		consumer.pending_action = None
		consumer.reschedule_details = None
		consumer._snooze_reminder = MagicMock(
			return_value=("Snoozed for 10 minutes.", "reminder_snoozed", True)
		)

		response, status, terminal = consumer._doctor_turn(
			"snooze me for 10 minutes",
			{},
			1,
		)

		self.assertEqual(response, "Snoozed for 10 minutes.")
		self.assertEqual(status, "reminder_snoozed")
		self.assertTrue(terminal)
		consumer._snooze_reminder.assert_called_once_with({}, 10)

	def test_reminder_snooze_never_calls_reschedule_and_keeps_appointment_time(self):
		appointment = self.create_appointment(
			start_time=self.now + timedelta(hours=1),
			end_time=self.now + timedelta(hours=1, minutes=30),
		)
		original_start = appointment.start_time
		snooze_started_at = django_timezone.now()
		reminder = ReminderService.create_for_appointment(appointment, now=self.now)
		consumer = object.__new__(DoctorVoiceConsumer)
		consumer.reminder_id = reminder.pk
		consumer.pending_action = None
		consumer.reschedule_details = None
		context = {
			"patient_name": appointment.caller_name,
			"clinic_tz": self.doctor.timezone,
		}

		with patch("core.doc_consumers.AppointmentService.reschedule") as reschedule:
			response, status, terminal = consumer._doctor_turn(
				"snooze for 10 minutes",
				context,
				1,
			)

		appointment.refresh_from_db()
		reminder.refresh_from_db()
		self.assertEqual(status, "reminder_snoozed")
		self.assertTrue(terminal)
		self.assertIn("remind you again in 10 minutes", response)
		self.assertEqual(appointment.start_time, original_start)
		self.assertLess(
			abs((reminder.next_fire_at - snooze_started_at - timedelta(minutes=10)).total_seconds()),
			2,
		)
		reschedule.assert_not_called()

	def test_incomplete_snooze_does_not_fall_through_to_rescheduling_model(self):
		consumer = object.__new__(DoctorVoiceConsumer)
		consumer.reminder_id = 17
		consumer.pending_action = None
		consumer.reschedule_details = None
		consumer._ask_doctor_model = MagicMock()

		response, status, terminal = consumer._doctor_turn("snooze 10", {}, 1)

		self.assertEqual(status, "snooze_details_needed")
		self.assertFalse(terminal)
		self.assertIn("number of minutes", response)
		consumer._ask_doctor_model.assert_not_called()

	def test_reminder_scheduler_claims_due_reminder_once_without_email(self):
		appointment = self.create_appointment(start_time=self.now + timedelta(minutes=10))
		reminder = ReminderService.create_for_appointment(appointment, now=self.now)
		reminder.next_fire_at = self.now
		reminder.save(update_fields=["next_fire_at"])

		self.assertTrue(claim_and_fire(reminder.pk, self.now))
		self.assertFalse(claim_and_fire(reminder.pk, self.now))
		reminder.refresh_from_db()
		self.assertEqual(reminder.status, Reminder.Status.ACTIVE)
		self.assertEqual(reminder.last_channel, "in_app")
		self.assertEqual(ReminderEvent.objects.filter(reminder=reminder, event_type="fired").count(), 1)

	@override_settings(DEBUG=True)
	def test_active_reminders_api_updates_presence_and_call_token(self):
		appointment = self.create_appointment(start_time=self.now + timedelta(minutes=10))
		reminder = ReminderService.create_for_appointment(appointment, now=self.now)
		reminder.status = Reminder.Status.ACTIVE
		reminder.save(update_fields=["status"])

		response = self.client.get("/api/doc/reminders/active/")

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.json()["active"][0]["reminder_id"], reminder.pk)
		self.doctor.refresh_from_db()
		self.assertIsNotNone(self.doctor.dashboard_last_seen_at)

		token_response = self.client.post(
			f"/api/doc/reminders/{reminder.pk}/call-token/",
		)
		self.assertEqual(token_response.status_code, 200)
		self.assertIn("token=", token_response.json()["ws_url"])

		assistant_token_response = self.client.post(
			"/api/doc/assistant/call-token/",
		)
		self.assertEqual(assistant_token_response.status_code, 200)
		self.assertIn("/ws/doc-assistant/?token=", assistant_token_response.json()["ws_url"])

	@override_settings(DEBUG=True)
	def test_snoozed_reminder_disappears_until_snooze_deadline(self):
		appointment = self.create_appointment(start_time=self.now + timedelta(hours=1))
		reminder = ReminderService.create_for_appointment(appointment, now=self.now)
		reminder.status = Reminder.Status.SNOOZED
		reminder.next_fire_at = self.now + timedelta(minutes=15)
		reminder.save(update_fields=["status", "next_fire_at"])

		response = self.client.get("/api/doc/reminders/active/")

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.json()["active"], [])

		reminder.next_fire_at = self.now - timedelta(seconds=1)
		reminder.save(update_fields=["next_fire_at"])

		response = self.client.get("/api/doc/reminders/active/")

		self.assertEqual(response.status_code, 200)
		active = response.json()["active"]
		self.assertEqual(len(active), 1)
		self.assertEqual(active[0]["reminder_id"], reminder.pk)
		self.assertEqual(active[0]["status"], Reminder.Status.ACTIVE)
		self.assertIsNone(active[0]["next_fire_at"])

	def test_appointment_service_finds_upcoming_by_patient_or_exact_time(self):
		mike = self.create_appointment(
			caller_name="Mike Smith",
			start_time=self.now + timedelta(hours=2),
			end_time=self.now + timedelta(hours=2, minutes=30),
		)
		self.create_appointment(
			caller_name="John Doe",
			start_time=self.now + timedelta(hours=3),
			end_time=self.now + timedelta(hours=3, minutes=30),
		)
		service = AppointmentService()

		by_name = service.find_upcoming(patient_name="mike")
		by_time = service.find_upcoming(start_time=mike.start_time.isoformat())

		self.assertEqual([item["appointment_id"] for item in by_name], [mike.pk])
		self.assertEqual([item["patient_name"] for item in by_time], ["Mike Smith"])

	def test_doctor_assistant_uses_database_results_before_answering(self):
		appointment = self.create_appointment(
			caller_name="Mike Smith",
			start_time=self.now + timedelta(hours=2),
			end_time=self.now + timedelta(hours=2, minutes=30),
		)
		consumer = object.__new__(DoctorAssistantConsumer)
		consumer.assistant_context = {
			"doctor_name": self.doctor.name,
			"clinic_tz": self.doctor.timezone,
		}
		consumer.known_appointment_ids = set()
		consumer.pending_action = None
		consumer.conversation_history = [
			{"role": "user", "content": "Do I have an appointment with Mike?"},
		]
		find_call = SimpleNamespace(
			id="lookup-1",
			function=SimpleNamespace(
				name="find_upcoming_appointments",
				arguments='{"patient_name":"Mike"}',
			),
		)
		first_message = SimpleNamespace(content=None, tool_calls=[find_call])
		final_message = SimpleNamespace(
			content="Yes, Mike Smith has an upcoming appointment.",
			tool_calls=None,
		)
		client = MagicMock()
		client.chat.completions.create.side_effect = [
			SimpleNamespace(choices=[SimpleNamespace(message=first_message)]),
			SimpleNamespace(choices=[SimpleNamespace(message=final_message)]),
		]
		consumer.llm = SimpleNamespace(client=client, model_name="test-model")

		response, status, terminal = consumer._assistant_turn(
			"Do I have an appointment with Mike?",
			2,
		)

		self.assertEqual(response, "Yes, Mike Smith has an upcoming appointment.")
		self.assertIsNone(status)
		self.assertFalse(terminal)
		self.assertIn(appointment.pk, consumer.known_appointment_ids)
		self.assertEqual(client.chat.completions.create.call_count, 2)
		followup_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
		tool_result = next(item for item in followup_messages if item.get("role") == "tool")
		self.assertIn("Mike Smith", tool_result["content"])

	def test_doctor_assistant_requires_confirmation_before_cancelling(self):
		appointment = self.create_appointment(
			caller_name="Mike Smith",
			start_time=self.now + timedelta(hours=2),
			end_time=self.now + timedelta(hours=2, minutes=30),
		)
		consumer = object.__new__(DoctorAssistantConsumer)
		consumer.assistant_context = {"clinic_tz": self.doctor.timezone}
		consumer.known_appointment_ids = {appointment.pk}
		consumer.pending_action = None
		proposal = consumer._run_assistant_tool(
			AppointmentService(),
			"propose_cancel",
			{"appointment_id": appointment.pk},
			2,
		)

		self.assertIn("Please say yes to confirm", proposal)
		appointment.refresh_from_db()
		self.assertEqual(appointment.status, "Scheduled")
		with patch("core.services.appointment_service.GoogleCalendarService") as calendar:
			calendar.return_value.is_authenticated.return_value = False
			response, status, terminal = consumer._assistant_turn("yes", 3)

		appointment.refresh_from_db()
		self.assertEqual(appointment.status, "Cancelled")
		self.assertEqual(status, "appointment_cancelled")
		self.assertIn("Cancelled Mike Smith", response)
		self.assertFalse(terminal)

	def test_doctor_assistant_negative_followup_ends_conversation(self):
		consumer = object.__new__(DoctorAssistantConsumer)
		consumer.assistant_context = {"clinic_tz": self.doctor.timezone}
		consumer.pending_action = None
		consumer.awaiting_goodbye = True

		response, status, terminal = consumer._assistant_turn("no, that's all", 2)

		self.assertEqual(response, "Thank you.")
		self.assertEqual(status, "conversation_complete")
		self.assertTrue(terminal)

	def test_doctor_assistant_requires_confirmation_before_rescheduling(self):
		appointment = self.create_appointment(
			caller_name="Mike Smith",
			start_time=self.now + timedelta(days=1),
			end_time=self.now + timedelta(days=1, minutes=30),
		)
		new_start = self.now + timedelta(days=2)
		consumer = object.__new__(DoctorAssistantConsumer)
		consumer.assistant_context = {"clinic_tz": self.doctor.timezone}
		consumer.known_appointment_ids = {appointment.pk}
		consumer.pending_action = None
		with patch("core.services.appointment_service.GoogleCalendarService") as calendar:
			calendar.return_value.is_authenticated.return_value = False
			proposal = consumer._run_assistant_tool(
				AppointmentService(),
				"propose_reschedule",
				{
					"appointment_id": appointment.pk,
					"new_start_time": new_start.isoformat(),
				},
				2,
			)
			self.assertIn("Please say yes to confirm", proposal)
			appointment.refresh_from_db()
			self.assertNotEqual(appointment.start_time, new_start)

			response, status, _ = consumer._assistant_turn("yes", 3)

		appointment.refresh_from_db()
		self.assertEqual(appointment.start_time, new_start)
		self.assertEqual(status, "appointment_rescheduled")
		self.assertIn("Rescheduled Mike Smith", response)

	@override_settings(DEBUG=False)
	def test_doctor_assistant_token_is_development_only(self):
		response = self.client.post("/api/doc/assistant/call-token/")

		self.assertEqual(response.status_code, 404)

	@override_settings(DEBUG=False)
	def test_reminder_endpoints_are_development_only(self):
		self.assertEqual(self.client.get("/api/doc/reminders/active/").status_code, 404)
		self.assertEqual(self.client.post("/api/doc/reminders/1/call-token/").status_code, 404)
		self.assertEqual(self.client.post("/api/doc/assistant/call-token/").status_code, 404)


class GoogleCalendarServiceTests(TransactionTestCase):
	def test_unauthenticated_raises_error(self):
		cal = GoogleCalendarService(token_file="nonexistent_token.json")
		with self.assertRaises(CalendarAuthError):
			cal.check_availability("2026-10-12T10:00:00Z", "2026-10-12T10:30:00Z")

	def test_check_availability_rejects_past_start_without_conflict_lookup(self):
		Doctor.objects.get_or_create(
			pk=Doctor.SINGLETON_ID,
			defaults={
				"name": "Dr. Melvin",
				"email": "doctor@example.test",
				"timezone": "Asia/Karachi",
			},
		)
		past_start = (django_timezone.now() - timedelta(minutes=1)).isoformat()
		with patch("core.services.appointment_service.Appointment.objects.filter") as appointment_filter:
			result = AppointmentService().check_availability(past_start)

		self.assertFalse(result["ok"])
		self.assertEqual(result["status"], "invalid_time")
		self.assertIn("future", result["message"])
		appointment_filter.assert_not_called()

	def test_check_availability_available(self):
		cal = GoogleCalendarService(token_file="nonexistent_token.json")
		mock_service = MagicMock()
		cal._service = mock_service

		mock_service.freebusy().query().execute.return_value = {
			'calendars': {
				'primary': {'busy': []}
			}
		}

		result = cal.check_availability("2026-10-12T10:00:00Z", "2026-10-12T10:30:00Z")
		self.assertTrue(result["available"])
		self.assertEqual(result["conflicts"], [])

	def test_check_availability_busy(self):
		cal = GoogleCalendarService(token_file="nonexistent_token.json")
		mock_service = MagicMock()
		cal._service = mock_service

		mock_service.freebusy().query().execute.return_value = {
			'calendars': {
				'primary': {'busy': [{'start': '2026-10-12T10:00:00Z', 'end': '2026-10-12T10:30:00Z'}]}
			}
		}
		mock_service.events().list().execute.return_value = {
			'items': [
				{
					'summary': 'Team Meeting',
					'start': {'dateTime': '2026-10-12T10:00:00Z'},
					'end': {'dateTime': '2026-10-12T10:30:00Z'}
				}
			]
		}

		result = cal.check_availability("2026-10-12T10:00:00Z", "2026-10-12T10:30:00Z")
		self.assertFalse(result["available"])
		self.assertEqual(len(result["conflicts"]), 1)
		self.assertEqual(result["conflicts"][0]["summary"], "Team Meeting")

	def test_create_event_success(self):
		cal = GoogleCalendarService(token_file="nonexistent_token.json")
		mock_service = MagicMock()
		cal._service = mock_service

		mock_service.events().insert().execute.return_value = {
			'id': 'test_event_123',
			'summary': 'Consultation with Alice',
			'htmlLink': 'https://calendar.google.com/event?eid=123',
			'start': {'dateTime': '2026-10-12T10:00:00Z'},
			'end': {'dateTime': '2026-10-12T10:30:00Z'},
			'status': 'confirmed'
		}

		created = cal.create_event(
			summary="Consultation with Alice",
			start_time="2026-10-12T10:00:00Z",
			end_time="2026-10-12T10:30:00Z",
			caller_name="Alice"
		)
		self.assertEqual(created["id"], "test_event_123")
		self.assertEqual(created["status"], "confirmed")


from core.services.vad_service import get_vad_service
from core.services.vad_service import VADService
from core.services.stt_service import STTService, get_stt_service
from core.services.tts_service import get_tts_service
from core.services.llm_service import LLMService
from core.models import Doctor, Appointment, Task


class VoiceOptimizationTests(TransactionTestCase):
	def test_transcription_uses_beam_size_one(self):
		service = STTService.__new__(STTService)
		service.language = "en"
		service.model = MagicMock()
		service.model.transcribe.return_value = ([MagicMock(text="test transcript")], None)

		audio_data = np.zeros(160, dtype=np.float32)
		result = service.transcribe(audio_data)

		self.assertEqual(result, "test transcript")
		service.model.transcribe.assert_called_once()
		self.assertEqual(
			service.model.transcribe.call_args.kwargs,
			{
				"beam_size": 1,
				"vad_filter": False,
				"condition_on_previous_text": False,
				"language": "en",
			},
		)

	def test_vad_trusts_silero_hub_repository(self):
		with patch(
			"core.services.vad_service.torch.hub.load",
			return_value=(object(), (None, None, None, None, None)),
		) as load_model:
			VADService()

		load_model.assert_called_once_with(
			repo_or_dir="snakers4/silero-vad",
			model="silero_vad",
			force_reload=False,
			trust_repo=True,
		)

	def test_service_singletons_return_same_instance(self):
		vad1 = get_vad_service()
		vad2 = get_vad_service()
		self.assertIs(vad1, vad2)

		stt1 = get_stt_service("base")
		stt2 = get_stt_service("base")
		self.assertIs(stt1, stt2)

		tts1 = get_tts_service()
		tts2 = get_tts_service()
		self.assertIs(tts1, tts2)


class LLMServiceToolsTests(TransactionTestCase):
	def setUp(self):
		self.doctor, _ = Doctor.objects.get_or_create(
			id=Doctor.SINGLETON_ID,
			defaults={"name": "Dr. Melvin", "email": "musa813705@gmail.com"}
		)

	def test_system_instruction_identifies_clinic_calendar(self):
		llm = LLMService()

		instruction = llm._get_system_instruction()

		self.assertIn("Dr. Melvin's clinic schedule", instruction)
		self.assertIn("Never claim that the appointment was added to the caller's personal calendar", instruction)
		self.assertIn("30 minutes by default", instruction)
		self.assertIn("pass new_end_time", instruction)
		self.assertIn("do not call availability or booking tools", instruction)
		self.assertIn("ask only for required details that are still missing", instruction)
		self.assertIn("caller_name and purpose", instruction)
		self.assertIn("call cancel_appointment directly", instruction)
		self.assertIn("Never invent or assume appointment details, availability, records, alternatives, or completed actions.", instruction)
		self.assertIn("ask the caller what other time they would prefer", instruction)
		self.assertIn("Do not suggest an unverified alternative.", instruction)
		self.assertIn("ask one concise clarifying question instead of guessing", instruction)
		self.assertIn("do not provide diagnoses or treatment advice", instruction)

	def test_availability_tool_description_does_not_claim_to_find_alternatives(self):
		availability_tool = next(
			tool["function"]
			for tool in LLMService.TOOL_DEFINITIONS
			if tool["function"]["name"] == "check_doctor_availability"
		)

		self.assertIn("does not find alternative open times", availability_tool["description"])
		self.assertIn("or check Google Calendar", availability_tool["description"])

	def test_past_appointment_is_rejected_before_lookup_or_booking(self):
		calendar = MagicMock()
		llm = LLMService(calendar_service=calendar)
		past_start = (django_timezone.now() - timedelta(minutes=1)).isoformat()

		with patch("core.services.llm_service.Appointment.objects.filter") as appointment_filter, patch(
			"core.services.llm_service.DataRepository.create_appointment_sync"
		) as create_appointment:
			availability = llm.check_doctor_availability(past_start)
			booking = llm.book_appointment(
				caller_name="Patient Example",
				start_time=past_start,
				purpose="Routine checkup",
			)

		self.assertFalse(availability["available"])
		self.assertEqual(availability["status"], "invalid_time")
		self.assertEqual(booking["status"], "invalid_time")
		self.assertIn("future", booking["message"])
		appointment_filter.assert_not_called()
		calendar.is_authenticated.assert_not_called()
		calendar.create_event.assert_not_called()
		create_appointment.assert_not_called()

	def test_scheduling_tool_schema_allows_default_duration_and_explicit_reschedule_end(self):
		tools = {
			tool["function"]["name"]: tool["function"]
			for tool in LLMService.TOOL_DEFINITIONS
		}

		self.assertEqual(
			tools["book_appointment"]["parameters"]["required"],
			["start_time"],
		)
		self.assertEqual(
			tools["check_doctor_availability"]["parameters"]["required"],
			["start_time"],
		)
		self.assertIn(
			"new_end_time",
			tools["reschedule_appointment"]["parameters"]["properties"],
		)
		self.assertNotIn(
			"new_end_time",
			tools["reschedule_appointment"]["parameters"]["required"],
		)
		self.assertEqual(
			tools["cancel_appointment"]["parameters"]["required"],
			["caller_name", "existing_start_time"],
		)

	def test_booking_details_are_carried_as_json_and_reused_after_asking_only_for_name(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = False
		llm = LLMService(calendar_service=mock_cal)
		start_time = "2026-10-15T21:00:00+05:00"
		llm.client = MagicMock()
		llm.client.chat.completions.create.side_effect = [
			FakeGroqStream([
				groq_tool_call_chunk(
					"availability-call",
					"check_doctor_availability",
					json.dumps({
						"start_time": start_time,
						"purpose": "Routine checkup",
					}),
				),
			]),
			FakeGroqStream([groq_content_chunk("What name should I put on the appointment?")]),
			FakeGroqStream([
				groq_tool_call_chunk(
					"booking-call",
					"book_appointment",
					json.dumps({
						"caller_name": "Mike Smith",
						"start_time": start_time,
					}),
				),
			]),
			FakeGroqStream([groq_content_chunk("Your routine checkup is booked for 9:00 PM.")]),
		]
		history = []

		first_reply, _ = llm.process_dialog(
			history,
			"I want to book an appointment at 9 PM today for a routine checkup.",
		)

		self.assertEqual(first_reply, "What name should I put on the appointment?")
		self.assertEqual(
			llm.booking_context["purpose"],
			"Routine checkup",
		)
		self.assertEqual(
			llm.booking_context["start_time"],
			"2026-10-15T21:00:00+05:00",
		)

		second_reply, status = llm.process_dialog(history, "My name is Mike Smith.")

		self.assertEqual(second_reply, "Your routine checkup is booked for 9:00 PM.")
		self.assertEqual(status, "calendar_booked")
		appointment = Appointment.objects.get(caller_name="Mike Smith")
		self.assertEqual(appointment.purpose, "Routine checkup")
		self.assertEqual(appointment.start_time, llm._parse_datetime(start_time))
		self.assertEqual(appointment.end_time - appointment.start_time, timedelta(minutes=30))
		self.assertEqual(llm.booking_context, {})

		second_booking_request = llm.client.chat.completions.create.call_args_list[2].kwargs["messages"]
		structured_context = next(
			message["content"]
			for message in second_booking_request
			if message["role"] == "system"
			and "structured details already extracted" in message["content"].lower()
		)
		self.assertIn('"purpose": "Routine checkup"', structured_context)

	def test_check_doctor_availability_uses_database_only_and_clinic_timezone(self):
		mock_cal = MagicMock()
		llm = LLMService(calendar_service=mock_cal)
		local_dt = django_timezone.localtime(
			django_timezone.now() + timedelta(hours=2)
		).replace(second=0, microsecond=0)
		end_dt = local_dt + timedelta(minutes=30)

		res = llm.check_doctor_availability(
			local_dt.isoformat(),
			end_dt.isoformat(),
		)
		self.assertTrue(res["available"])
		self.assertEqual(res["conflicts"], [])
		self.assertIn(
			f"{local_dt.strftime('%I:%M %p')} and {end_dt.strftime('%I:%M %p')}",
			res["message"],
		)
		mock_cal.is_authenticated.assert_not_called()
		mock_cal.check_availability.assert_not_called()

		# Naive tool datetimes are clinic-local; offset datetimes are converted to clinic-local.
		parsed_local_dt = llm._parse_datetime(local_dt.replace(tzinfo=None).isoformat())
		offset_dt = llm._parse_datetime(
			local_dt.astimezone(datetime_timezone.utc).isoformat()
		)
		self.assertEqual(parsed_local_dt, local_dt)
		self.assertEqual(offset_dt, local_dt)

		# Book an existing appointment in DB
		Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Existing Patient",
			start_time=local_dt,
			end_time=end_dt,
			purpose="Checkup",
			status="Scheduled"
		)

		# Conflicting slot
		res2 = llm.check_doctor_availability(
			local_dt.isoformat(),
			end_dt.isoformat(),
		)
		self.assertFalse(res2["available"])
		self.assertEqual(llm.last_tool_status, "conflict_detected")
		self.assertIn(
			f"{local_dt.strftime('%I:%M %p')} to {end_dt.strftime('%I:%M %p')}",
			res2["message"],
		)
		mock_cal.is_authenticated.assert_not_called()
		mock_cal.check_availability.assert_not_called()

	def test_book_appointment_persists_to_db_without_creating_chart_review_task(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = True
		mock_cal.create_event.return_value = {"id": "google-event-123"}
		llm = LLMService(calendar_service=mock_cal)
		start_time = django_timezone.localtime(
			django_timezone.now() + timedelta(days=2)
		).replace(hour=10, minute=0, second=0, microsecond=0)

		result = llm.book_appointment(
			caller_name="John Doe",
			start_time=start_time.isoformat(),
			end_time=(start_time + timedelta(minutes=30)).isoformat(),
			purpose="General Consultation"
		)

		self.assertEqual(result["status"], "confirmed")
		self.assertEqual(result["caller_name"], "John Doe")
		self.assertEqual(llm.last_tool_status, "calendar_booked")
		self.assertTrue(result["google_calendar_synced"])
		calendar_start = mock_cal.create_event.call_args.kwargs["start_time"]
		self.assertEqual(calendar_start, start_time)

		# Verify DB records
		app = Appointment.objects.get(id=result["appointment_id"])
		self.assertEqual(app.caller_name, "John Doe")
		self.assertEqual(app.purpose, "General Consultation")
		self.assertEqual(app.start_time, start_time)
		self.assertEqual(
			result["start_time"],
			start_time.strftime("%A, %B %d at %I:%M %p"),
		)

		self.assertFalse(Task.objects.filter(appointment=app).exists())
		self.assertEqual(app.google_event_id, "google-event-123")

	def test_book_appointment_defaults_to_thirty_minutes(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = False
		llm = LLMService(calendar_service=mock_cal)

		result = llm.book_appointment(
			caller_name="Mike Smith",
			start_time=(django_timezone.now() + timedelta(days=2)).isoformat(),
			purpose="Checkup",
		)

		self.assertEqual(result["status"], "confirmed")
		self.assertFalse(result["google_calendar_synced"])
		self.assertEqual(llm.last_tool_status, "calendar_sync_failed")
		self.assertIn("confirmed in the clinic scheduling system", result["message"])
		self.assertNotIn("Google Calendar", result["message"])
		self.assertFalse(mock_cal.create_event.called)
		appointment = Appointment.objects.get(pk=result["appointment_id"])
		self.assertEqual(
			appointment.end_time - appointment.start_time,
			timedelta(minutes=30),
		)
		self.assertIsNone(appointment.google_event_id)

	def test_book_appointment_reports_calendar_sync_exception(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = True
		mock_cal.create_event.side_effect = RuntimeError("Calendar API unavailable")
		llm = LLMService(calendar_service=mock_cal)

		result = llm.book_appointment(
			caller_name="Mike Smith",
			start_time=(django_timezone.now() + timedelta(days=2)).isoformat(),
			purpose="Checkup",
		)

		self.assertEqual(result["status"], "confirmed")
		self.assertFalse(result["google_calendar_synced"])
		self.assertEqual(llm.last_tool_status, "calendar_sync_failed")
		self.assertIn("confirmed in the clinic scheduling system", result["message"])
		self.assertNotIn("Google Calendar", result["message"])
		appointment = Appointment.objects.get(pk=result["appointment_id"])
		self.assertIsNone(appointment.google_event_id)

	def test_availability_defaults_to_thirty_minute_window(self):
		llm = LLMService(calendar_service=MagicMock())
		start_time = django_timezone.now() + timedelta(days=2)
		start_time = start_time.replace(second=0, microsecond=0)

		result = llm.check_doctor_availability(start_time.isoformat())

		self.assertTrue(result["available"])
		appointment_start = start_time + timedelta(minutes=20)
		Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Other Patient",
			start_time=appointment_start,
			end_time=appointment_start + timedelta(minutes=10),
			purpose="Conflict",
		)
		conflict_result = llm.check_doctor_availability(start_time.isoformat())
		self.assertFalse(conflict_result["available"])

	def test_list_upcoming_appointments_matches_name_and_only_scheduled_future_records(self):
		llm = LLMService(calendar_service=MagicMock())
		start_time = django_timezone.now() + timedelta(days=2)
		appointment = Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Mike Smith",
			start_time=start_time,
			end_time=start_time + timedelta(minutes=30),
			purpose="Checkup",
		)
		Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Mike Smith",
			start_time=start_time + timedelta(days=1),
			end_time=start_time + timedelta(days=1, minutes=30),
			purpose="Cancelled visit",
			status="Cancelled",
		)

		result = llm.list_upcoming_appointments(" mike smith ")

		self.assertEqual(result["status"], "found")
		self.assertEqual([item["appointment_id"] for item in result["appointments"]], [appointment.pk])
		self.assertEqual(result["appointments"][0]["purpose"], "Checkup")
		self.assertEqual(llm.last_tool_status, "appointments_found")

	def test_reschedule_updates_database_calendar_and_reminder(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = True
		llm = LLMService(calendar_service=mock_cal)
		old_start = django_timezone.now() + timedelta(days=2)
		old_start = old_start.replace(second=0, microsecond=0)
		old_end = old_start + timedelta(minutes=45)
		appointment = Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Mike Smith",
			start_time=old_start,
			end_time=old_end,
			purpose="Checkup",
			google_event_id="google-event-456",
		)
		new_start = old_start + timedelta(days=3)
		new_start = new_start.replace(hour=23, minute=0)
		new_end = new_start + timedelta(hours=1)

		result = llm.reschedule_appointment(
			caller_name="mike smith",
			existing_start_time=old_start.isoformat(),
			new_start_time=new_start.isoformat(),
			new_end_time=new_end.isoformat(),
		)

		appointment.refresh_from_db()
		self.assertEqual(result["status"], "updated")
		self.assertEqual(llm.last_tool_status, "appointment_rescheduled")
		self.assertEqual(appointment.start_time, new_start)
		self.assertEqual(appointment.end_time, new_end)
		reminder = Reminder.objects.get(appointment=appointment)
		self.assertEqual(reminder.status, Reminder.Status.SCHEDULED)
		self.assertEqual(reminder.next_fire_at, new_start - timedelta(minutes=30))
		mock_cal.update_event.assert_called_once_with(
			event_id="google-event-456",
			start_time=new_start,
			end_time=new_end,
		)

	def test_reschedule_defaults_to_existing_duration_when_end_is_not_requested(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = False
		llm = LLMService(calendar_service=mock_cal)
		old_start = django_timezone.now() + timedelta(days=2)
		old_start = old_start.replace(second=0, microsecond=0)
		appointment = Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Mike Smith",
			start_time=old_start,
			end_time=old_start + timedelta(minutes=45),
			purpose="Checkup",
		)
		new_start = old_start + timedelta(days=1)

		result = llm.reschedule_appointment(
			caller_name="Mike Smith",
			existing_start_time=old_start.isoformat(),
			new_start_time=new_start.isoformat(),
		)

		appointment.refresh_from_db()
		self.assertEqual(result["status"], "updated")
		self.assertEqual(appointment.end_time, new_start + timedelta(minutes=45))

	def test_reschedule_rejects_end_not_after_new_start(self):
		mock_cal = MagicMock()
		llm = LLMService(calendar_service=mock_cal)
		old_start = django_timezone.now() + timedelta(days=2)
		old_start = old_start.replace(second=0, microsecond=0)
		Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Mike Smith",
			start_time=old_start,
			end_time=old_start + timedelta(minutes=30),
			purpose="Checkup",
		)
		new_start = old_start + timedelta(days=1)

		result = llm.reschedule_appointment(
			caller_name="Mike Smith",
			existing_start_time=old_start.isoformat(),
			new_start_time=new_start.isoformat(),
			new_end_time=(new_start - timedelta(minutes=1)).isoformat(),
		)

		self.assertEqual(result["status"], "error")
		self.assertEqual(llm.last_tool_status, "error")

	def test_reschedule_rejects_conflict_without_updating_calendar_or_appointment(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = True
		llm = LLMService(calendar_service=mock_cal)
		old_start = django_timezone.now() + timedelta(days=2)
		old_start = old_start.replace(second=0, microsecond=0)
		appointment = Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Mike Smith",
			start_time=old_start,
			end_time=old_start + timedelta(minutes=30),
			purpose="Checkup",
			google_event_id="google-event-456",
		)
		conflict_start = old_start + timedelta(days=1)
		Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Other Patient",
			start_time=conflict_start,
			end_time=conflict_start + timedelta(minutes=30),
			purpose="Existing visit",
		)

		result = llm.reschedule_appointment(
			caller_name="Mike Smith",
			existing_start_time=old_start.isoformat(),
			new_start_time=conflict_start.isoformat(),
		)

		self.assertEqual(result["status"], "conflict")
		self.assertEqual(llm.last_tool_status, "conflict_detected")
		appointment.refresh_from_db()
		self.assertEqual(appointment.start_time, old_start)
		mock_cal.update_event.assert_not_called()

	def test_reschedule_calendar_failure_does_not_update_database(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = True
		mock_cal.update_event.side_effect = RuntimeError("Calendar unavailable")
		llm = LLMService(calendar_service=mock_cal)
		old_start = django_timezone.now() + timedelta(days=2)
		old_start = old_start.replace(second=0, microsecond=0)
		appointment = Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Mike Smith",
			start_time=old_start,
			end_time=old_start + timedelta(minutes=30),
			purpose="Checkup",
			google_event_id="google-event-456",
		)

		result = llm.reschedule_appointment(
			caller_name="Mike Smith",
			existing_start_time=old_start.isoformat(),
			new_start_time=(old_start + timedelta(days=1)).isoformat(),
		)

		self.assertEqual(result["status"], "error")
		self.assertEqual(llm.last_tool_status, "error")
		appointment.refresh_from_db()
		self.assertEqual(appointment.start_time, old_start)

	def test_calendar_update_patches_only_event_start_and_end(self):
		mock_resource = MagicMock()
		mock_resource.events.return_value.patch.return_value.execute.return_value = {
			"id": "google-event-456",
			"start": {"dateTime": "2026-10-20T15:00:00+05:00"},
			"end": {"dateTime": "2026-10-20T15:30:00+05:00"},
			"status": "confirmed",
		}
		calendar = GoogleCalendarService()
		calendar.get_service = MagicMock(return_value=mock_resource)
		start_time = LLMService()._parse_datetime("2026-10-20 15:00")
		end_time = start_time + timedelta(minutes=30)

		result = calendar.update_event("google-event-456", start_time, end_time)

		self.assertEqual(result["id"], "google-event-456")
		mock_resource.events.return_value.patch.assert_called_once_with(
			calendarId="primary",
			eventId="google-event-456",
			body={
				"start": {"dateTime": start_time.isoformat()},
				"end": {"dateTime": end_time.isoformat()},
			},
		)

	def test_cancel_appointment_cancels_database_calendar_and_reminder(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = True
		llm = LLMService(calendar_service=mock_cal)
		start_time = django_timezone.now() + timedelta(days=2)
		start_time = start_time.replace(second=0, microsecond=0)
		appointment = Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Mike Smith",
			start_time=start_time,
			end_time=start_time + timedelta(minutes=30),
			purpose="Checkup",
			google_event_id="google-event-cancel-1",
		)
		result = llm.cancel_appointment("mike smith", start_time.isoformat())

		appointment.refresh_from_db()
		self.assertEqual(result["status"], "cancelled")
		self.assertEqual(llm.last_tool_status, "appointment_cancelled")
		self.assertEqual(appointment.status, "Cancelled")
		mock_cal.delete_event.assert_called_once_with(event_id="google-event-cancel-1")

	def test_cancel_appointment_calendar_failure_keeps_database_scheduled(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = True
		mock_cal.delete_event.side_effect = RuntimeError("Calendar unavailable")
		llm = LLMService(calendar_service=mock_cal)
		start_time = django_timezone.now() + timedelta(days=2)
		start_time = start_time.replace(second=0, microsecond=0)
		appointment = Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Mike Smith",
			start_time=start_time,
			end_time=start_time + timedelta(minutes=30),
			purpose="Checkup",
			google_event_id="google-event-cancel-2",
		)

		result = llm.cancel_appointment("Mike Smith", start_time.isoformat())

		appointment.refresh_from_db()
		self.assertEqual(result["status"], "error")
		self.assertEqual(appointment.status, "Scheduled")

	def test_cancel_appointment_requires_unique_name_and_start_time_match(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = True
		llm = LLMService(calendar_service=mock_cal)
		start_time = django_timezone.now() + timedelta(days=2)
		start_time = start_time.replace(second=0, microsecond=0)
		appointments = [
			Appointment.objects.create(
				doctor=self.doctor,
				caller_name="Mike Smith",
				start_time=start_time,
				end_time=start_time + timedelta(minutes=30),
				purpose=f"Visit {index}",
				google_event_id=f"google-event-cancel-{index}",
			)
			for index in range(2)
		]

		result = llm.cancel_appointment("Mike Smith", start_time.isoformat())

		self.assertEqual(result["status"], "ambiguous")
		self.assertEqual(llm.last_tool_status, "appointment_ambiguous")
		for appointment in appointments:
			appointment.refresh_from_db()
			self.assertEqual(appointment.status, "Scheduled")
		mock_cal.delete_event.assert_not_called()

	def test_calendar_delete_uses_the_saved_event_id(self):
		mock_resource = MagicMock()
		calendar = GoogleCalendarService()
		calendar.get_service = MagicMock(return_value=mock_resource)

		calendar.delete_event("google-event-cancel-3")

		mock_resource.events.return_value.delete.assert_called_once_with(
			calendarId="primary",
			eventId="google-event-cancel-3",
		)

	def test_process_dialog_fallback_on_error(self):
		llm = LLMService()
		llm.client = MagicMock()
		llm.client.chat.completions.create.side_effect = Exception("Simulated API error")

		reply, tool = llm.process_dialog([], "test hello")
		self.assertEqual(reply, "I am having trouble connecting to my scheduling system right now. Could you please repeat that?")
		self.assertEqual(tool, "error")

	def test_process_dialog_success(self):
		llm = LLMService()
		llm.client = MagicMock()
		llm.client.chat.completions.create.return_value = FakeGroqStream([
			groq_content_chunk("Hello! "),
			groq_content_chunk("Dr. Melvin is available."),
		])

		history = []
		with self.assertLogs("core.services.llm_service", level="INFO") as timing_logs:
			reply, tool = llm.process_dialog(history, "hello doctor")

		self.assertTrue(any(
			"Pipeline Groq: response stream completed model=" in entry and "elapsed_ms=" in entry
			for entry in timing_logs.output
		))
		self.assertTrue(llm.client.chat.completions.create.call_args.kwargs["stream"])
		self.assertEqual(
			llm.client.chat.completions.create.call_args.kwargs["max_completion_tokens"],
			280,
		)
		self.assertEqual(
			llm.client.chat.completions.create.call_args.kwargs["reasoning_effort"],
			"low",
		)
		self.assertEqual(reply, "Hello! Dr. Melvin is available.")
		self.assertEqual(len(history), 2)
		self.assertEqual(history[0]["role"], "user")
		self.assertEqual(history[1]["role"], "assistant")

	def test_process_dialog_executes_tool_and_sends_result_back(self):
		llm = LLMService()
		llm.client = MagicMock()
		arguments = json.dumps({
			"start_time": "2026-10-15T14:00:00Z",
			"end_time": "2026-10-15T14:30:00Z",
		})
		llm.client.chat.completions.create.side_effect = [
			FakeGroqStream([
				groq_tool_call_chunk(
					"call-123",
					"check_doctor_availability",
					arguments,
				),
			]),
			FakeGroqStream([groq_content_chunk("That time is available.")]),
		]
		llm.check_doctor_availability = MagicMock(return_value={"available": True})

		history = []
		reply, tool = llm.process_dialog(history, "Is the doctor free then?")

		self.assertEqual(reply, "That time is available.")
		llm.check_doctor_availability.assert_called_once_with(
			start_time="2026-10-15T14:00:00Z",
			end_time="2026-10-15T14:30:00Z",
		)
		second_request_messages = llm.client.chat.completions.create.call_args_list[1].kwargs["messages"]
		first_request = llm.client.chat.completions.create.call_args_list[0].kwargs
		self.assertEqual(first_request["max_completion_tokens"], 280)
		self.assertEqual(first_request["tool_choice"], "auto")
		self.assertEqual(
			[
				tool["function"]["name"]
				for tool in first_request["tools"]
			],
			[
				"check_doctor_availability",
				"book_appointment",
				"list_upcoming_appointments",
				"reschedule_appointment",
				"cancel_appointment",
			],
		)
		self.assertEqual(second_request_messages[-1]["role"], "tool")
		self.assertEqual(second_request_messages[-1]["tool_call_id"], "call-123")
		self.assertEqual(json.loads(second_request_messages[-1]["content"]), {"available": True})
		self.assertEqual(tool, None)
		self.assertEqual(len(history), 2)
