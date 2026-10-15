import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'voice_agent_project.settings')

from django.core.asgi import get_asgi_application

django_asgi_app = get_asgi_application()

from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator
from channels.routing import ProtocolTypeRouter, URLRouter
from core.routing import websocket_urlpatterns

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        AllowedHostsOriginValidator(URLRouter(websocket_urlpatterns))
    ),
})