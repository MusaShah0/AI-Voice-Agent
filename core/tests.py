import base64
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from channels.db import database_sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.test import TransactionTestCase

from .models import Transcript
from .routing import websocket_urlpatterns


class FakeVADService:
	def __init__(self):
		self.call_count = 0

	def is_speech(self, audio_chunk):
		# First call returns True (speech start), subsequent calls return False (silence endpoint)
		self.call_count += 1
		return self.call_count == 1


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
			self.assertEqual(response["turn_id"], 1)
			self.assertEqual(
				self.fake_tts.inputs,
				["Dr. Melvin is available.", "Would you like to book?"],
			)
			self.assertTrue(stream_messages)
			self.assertTrue(all(message["type"] == "audio_chunk" for message in stream_messages))
			self.assertEqual(
				b"".join(base64.b64decode(message["audio_payload"]) for message in stream_messages),
				b"test audio" * len(self.fake_tts.inputs),
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

	async def test_binary_pcm_is_converted_to_float32_for_transcription(self):
		await self.connect_communicator()
		try:
			# Speech chunk (16,000 bytes = 0.5s speech)
			speech_chunk = b"\x01\x00" * 8000
			# Silence chunk (16,000 bytes = 0.5s silence, exactly the endpoint threshold)
			silence_chunk = b"\x00\x00" * 8000
			await self.communicator.send_to(bytes_data=speech_chunk)
			await self.communicator.send_to(bytes_data=silence_chunk)

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
			self.assertEqual(FakeSTTService.last_audio_parameters, (np.dtype(np.float32), 16000))
		finally:
			await self.communicator.disconnect()



from unittest.mock import MagicMock
from core.services.calendar_service import GoogleCalendarService, CalendarAuthError


class GoogleCalendarServiceTests(TransactionTestCase):
	def test_unauthenticated_raises_error(self):
		cal = GoogleCalendarService(token_file="nonexistent_token.json")
		with self.assertRaises(CalendarAuthError):
			cal.check_availability("2026-10-12T10:00:00Z", "2026-10-12T10:30:00Z")

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

	def test_check_doctor_availability_without_calendar(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = False
		llm = LLMService(calendar_service=mock_cal)

		# Free slot
		res = llm.check_doctor_availability("2026-10-15 14:00:00", "2026-10-15 14:30:00")
		self.assertTrue(res["available"])
		self.assertEqual(res["conflicts"], [])

		# Book an existing appointment in DB
		Appointment.objects.create(
			doctor=self.doctor,
			caller_name="Existing Patient",
			start_time="2026-10-15 14:00:00+00:00",
			end_time="2026-10-15 14:30:00+00:00",
			purpose="Checkup",
			status="Scheduled"
		)

		# Conflicting slot
		res2 = llm.check_doctor_availability("2026-10-15 14:00:00", "2026-10-15 14:30:00")
		self.assertFalse(res2["available"])
		self.assertEqual(llm.last_tool_status, "conflict_detected")

	def test_book_appointment_persists_to_db_and_creates_task(self):
		mock_cal = MagicMock()
		mock_cal.is_authenticated.return_value = False
		llm = LLMService(calendar_service=mock_cal)

		result = llm.book_appointment(
			caller_name="John Doe",
			start_time="2026-10-16 10:00:00",
			end_time="2026-10-16 10:30:00",
			purpose="General Consultation"
		)

		self.assertEqual(result["status"], "confirmed")
		self.assertEqual(result["caller_name"], "John Doe")
		self.assertEqual(llm.last_tool_status, "calendar_booked")

		# Verify DB records
		app = Appointment.objects.get(id=result["appointment_id"])
		self.assertEqual(app.caller_name, "John Doe")
		self.assertEqual(app.purpose, "General Consultation")

		# Verify task created
		task = Task.objects.get(appointment=app)
		self.assertIn("John Doe", task.description)

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
			["check_doctor_availability", "book_appointment"],
		)
		self.assertEqual(second_request_messages[-1]["role"], "tool")
		self.assertEqual(second_request_messages[-1]["tool_call_id"], "call-123")
		self.assertEqual(json.loads(second_request_messages[-1]["content"]), {"available": True})
		self.assertEqual(tool, None)
		self.assertEqual(len(history), 2)
