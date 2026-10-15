from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

SECRET_KEY = 'django-insecure-local-dev-key-change-me'
DEBUG = os.environ.get('DEBUG', 'true').lower() == 'true'
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'daphne',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'channels',
    'core',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'voice_agent_project.urls'

# --- ADD THIS TEMPLATES BLOCK TO FIX ADMIN ERROR ---
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'voice_agent_project.wsgi.application'
ASGI_APPLICATION = 'voice_agent_project.asgi.application'

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer"
    }
}

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# --- FIXES PRIMARY KEY WARNING ---
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Karachi'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'

REMINDER_LEAD_MINUTES = int(os.environ.get('REMINDER_LEAD_MINUTES', '30'))
REMINDER_REPEAT_MINUTES = int(os.environ.get('REMINDER_REPEAT_MINUTES', '5'))
SNOOZE_MINUTES = int(os.environ.get('SNOOZE_MINUTES', '15'))
SNOOZE_MIN_BUFFER_MINUTES = int(os.environ.get('SNOOZE_MIN_BUFFER_MINUTES', '1'))
REMINDER_MAX_ATTEMPTS = int(os.environ.get('REMINDER_MAX_ATTEMPTS', '6'))
REMINDER_MIN_LEAD_TO_REMIND_MINUTES = int(
    os.environ.get('REMINDER_MIN_LEAD_TO_REMIND_MINUTES', '2')
)
DOCTOR_PRESENCE_TTL_SECONDS = int(os.environ.get('DOCTOR_PRESENCE_TTL_SECONDS', '30'))
VOICE_INCLUDE_PURPOSE = os.environ.get('VOICE_INCLUDE_PURPOSE', 'true').lower() == 'true'
CALL_AUTO_HANGUP_AFTER_TERMINAL = (
    os.environ.get('CALL_AUTO_HANGUP_AFTER_TERMINAL', 'true').lower() == 'true'
)
DOCTOR_GROQ_MODEL = os.environ.get('DOCTOR_GROQ_MODEL', '').strip()

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'pipeline': {
            'format': '{asctime} {levelname} {name}: {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'pipeline',
            'level': 'INFO',
        },
    },
    'loggers': {
        'core': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}