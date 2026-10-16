"""
WSGI config for voice_agent_project project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'voice_agent_project.settings')

application = get_wsgi_application()

from core.reminders.scheduler import start_reminder_scheduler

start_reminder_scheduler()
