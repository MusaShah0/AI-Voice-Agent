from django.urls import re_path
from . import consumers, doc_consumers

websocket_urlpatterns = [
    re_path(r'ws/voice/$', consumers.VoiceConsumer.as_asgi()),
    re_path(r'ws/doc-voice/$', doc_consumers.DoctorVoiceConsumer.as_asgi()),
    re_path(r'ws/doc-assistant/$', doc_consumers.DoctorAssistantConsumer.as_asgi()),
]