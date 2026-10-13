import base64
import json
import wave
from unittest.mock import patch

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

	def __init__(self, model_size):
		pass

	def transcribe(self, audio_file_path):
		with wave.open(audio_file_path, "rb") as audio_file:
			type(self).last_audio_parameters = (
				audio_file.getnchannels(),
				audio_file.getsampwidth(),
				audio_file.getframerate(),
			)
		return "test transcription"


class FakeLLMService:
	def process_dialog(self, conversation_history, user_input):
		return "test response", None


class FakeTTSService:
	def synthesize(self, text):
		return b"test audio"


class VoiceConsumerTests(TransactionTestCase):
	def setUp(self):
		self.fake_vad = FakeVADService()
		self.patches = [
			patch("core.consumers.VADService", return_value=self.fake_vad),
			patch("core.consumers.STTService", return_value=FakeSTTService("base")),
			patch("core.consumers.LLMService", return_value=FakeLLMService()),
			patch("core.consumers.TTSService", return_value=FakeTTSService()),
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

				response = await self.communicator.receive_json_from(timeout=3)

			log_output = "\n".join(pipeline_logs.output)
			for stage in (
				"Pipeline receive: action=process_text_turn",
				"Pipeline dialog: started",
				"Pipeline dialog: completed",
				"Pipeline TTS: synthesis started",
				"Pipeline persistence: transcript saved",
				"Pipeline delivery: response sent",
			):
				self.assertIn(stage, log_output)
			self.assertNotIn("hello", log_output)

			self.assertEqual(response["type"], "agent_response")
			self.assertEqual(response["caption"], "test response")
			self.assertEqual(base64.b64decode(response["audio_payload"]), b"test audio")
			transcript = await database_sync_to_async(
				lambda: Transcript.objects.get(session_id__startswith="session_")
			)()
			self.assertEqual(transcript.full_transcript, "User: hello | Agent: test response")
		finally:
			await self.communicator.disconnect()

	async def test_binary_pcm_is_wrapped_as_wav_before_transcription(self):
		await self.connect_communicator()
		try:
			# Speech chunk (16,000 bytes = 0.5s speech)
			speech_chunk = b"\x01\x00" * 8000
			# Silence chunk (24,000 bytes = 0.75s silence, triggers SILENCE_THRESHOLD_BYTES)
			silence_chunk = b"\x00\x00" * 12000
			await self.communicator.send_to(bytes_data=speech_chunk)
			await self.communicator.send_to(bytes_data=silence_chunk)

			response = await self.communicator.receive_json_from(timeout=3)

			self.assertEqual(response["user_transcription"], "test transcription")
			self.assertEqual(FakeSTTService.last_audio_parameters, (1, 2, 16000))
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
from core.services.stt_service import get_stt_service
from core.services.tts_service import get_tts_service
from core.services.llm_service import LLMService
from core.models import Doctor, Appointment, Task


class VoiceOptimizationTests(TransactionTestCase):
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
		mock_chat = MagicMock()
		mock_chat.send_message.side_effect = Exception("Simulated API 429 Quota Error")
		llm._chat = mock_chat

		# Mock client so fallback also raises
		mock_client = MagicMock()
		mock_client.chats.create.side_effect = Exception("Simulated Secondary Error")
		llm.client = mock_client

		reply, tool = llm.process_dialog([], "test hello")
		self.assertEqual(reply, "I am having trouble connecting to my scheduling system right now. Could you please repeat that?")
		self.assertEqual(tool, "error")

	def test_process_dialog_success(self):
		llm = LLMService()
		mock_chat = MagicMock()
		mock_response = MagicMock()
		mock_response.text = "Hello! Dr. Melvin is available."
		mock_chat.send_message.return_value = mock_response
		llm._chat = mock_chat

		history = []
		reply, tool = llm.process_dialog(history, "hello doctor")
		self.assertEqual(reply, "Hello! Dr. Melvin is available.")
		self.assertEqual(len(history), 2)
		self.assertEqual(history[0]["role"], "user")
		self.assertEqual(history[1]["role"], "assistant")
