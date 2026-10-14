import glob
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/calendar']


class CalendarAuthError(Exception):
    """Raised when Google Calendar authentication is missing or invalid."""
    pass


class GoogleCalendarService:
    """
    Manages Google Calendar API integration for checking appointment availability
    and creating scheduled calendar events.
    """

    def __init__(
        self,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
        base_dir: Optional[Path] = None
    ):
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.credentials_file = credentials_file or self._find_client_secret_file()
        self.token_file = token_file or str(self.base_dir / 'token.json')
        self._service = None

    def _find_client_secret_file(self) -> Optional[str]:
        """Locates client_secret_*.json or credentials.json in base directory."""
        candidates = glob.glob(str(self.base_dir / 'client_secret*.json'))
        if candidates:
            return candidates[0]

        default_file = self.base_dir / 'credentials.json'
        if default_file.exists():
            return str(default_file)

        return None

    def get_credentials(self) -> Optional[Credentials]:
        """Loads or refreshes cached credentials from token.json."""
        creds = None
        if os.path.exists(self.token_file):
            try:
                creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
            except Exception as e:
                logger.warning("Failed to load existing token file: %s", e)
                creds = None

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                with open(self.token_file, 'w', encoding='utf-8') as token_out:
                    token_out.write(creds.to_json())
            except Exception as e:
                logger.warning("Failed to refresh Google Calendar credentials: %s", e)
                creds = None

        return creds

    def is_authenticated(self) -> bool:
        """Checks if a valid, unexpired credential set is available."""
        creds = self.get_credentials()
        return bool(creds and creds.valid)

    def authenticate_interactive(self, port: int = 0) -> bool:
        """
        Runs an interactive OAuth flow in the user's browser and saves token.json.
        Can be invoked via CLI: python -m core.services.calendar_service
        """
        if not self.credentials_file or not os.path.exists(self.credentials_file):
            raise CalendarAuthError(
                f"Client secret JSON file not found in {self.base_dir}. "
                "Download it from Google Cloud Console as client_secret_*.json"
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            self.credentials_file,
            scopes=SCOPES
        )
        creds = flow.run_local_server(port=port)
        with open(self.token_file, 'w', encoding='utf-8') as token_out:
            token_out.write(creds.to_json())

        self._service = None
        logger.info("Successfully authenticated Google Calendar and saved token to %s", self.token_file)
        return True

    def get_service(self):
        """Constructs and returns the Google Calendar API Resource."""
        if self._service is not None:
            return self._service

        creds = self.get_credentials()
        if not creds or not creds.valid:
            raise CalendarAuthError(
                "Google Calendar is not authenticated. Please run the OAuth authorization flow."
            )

        self._service = build('calendar', 'v3', credentials=creds)
        return self._service

    @staticmethod
    def _to_rfc3339(dt: Any) -> str:
        """Converts datetime or ISO string to RFC3339 string format required by Google Calendar."""
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.isoformat()

    def check_availability(
        self,
        start_time: Any,
        end_time: Any,
        calendar_id: str = 'primary'
    ) -> Dict[str, Any]:
        """
        Queries freebusy status for the specified calendar between start_time and end_time.
        Returns:
            {
                "available": bool,
                "start_time": str,
                "end_time": str,
                "conflicts": List[Dict],
            }
        """
        start_rfc = self._to_rfc3339(start_time)
        end_rfc = self._to_rfc3339(end_time)

        service = self.get_service()

        # 1. Query freebusy API
        body = {
            "timeMin": start_rfc,
            "timeMax": end_rfc,
            "items": [{"id": calendar_id}]
        }

        freebusy_result = service.freebusy().query(body=body).execute()
        calendars = freebusy_result.get('calendars', {})
        calendar_info = calendars.get(calendar_id, {})
        busy_slots = calendar_info.get('busy', [])

        if not busy_slots:
            return {
                "available": True,
                "start_time": start_rfc,
                "end_time": end_rfc,
                "conflicts": []
            }

        # 2. Retrieve detailed conflict summaries if busy
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=start_rfc,
            timeMax=end_rfc,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        conflicts = []
        for item in events_result.get('items', []):
            conflicts.append({
                "summary": item.get('summary', 'Busy'),
                "start": item.get('start', {}).get('dateTime', item.get('start', {}).get('date')),
                "end": item.get('end', {}).get('dateTime', item.get('end', {}).get('date')),
            })

        return {
            "available": False,
            "start_time": start_rfc,
            "end_time": end_rfc,
            "conflicts": conflicts
        }

    def create_event(
        self,
        summary: str,
        start_time: Any,
        end_time: Any,
        description: str = "",
        calendar_id: str = 'primary',
        caller_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a confirmed event on Google Calendar.
        """
        start_rfc = self._to_rfc3339(start_time)
        end_rfc = self._to_rfc3339(end_time)

        service = self.get_service()

        event_body = {
            'summary': summary,
            'description': description or f"Appointment booked via Voice Agent for {caller_name or 'Caller'}",
            'start': {'dateTime': start_rfc},
            'end': {'dateTime': end_rfc},
        }

        created_event = service.events().insert(
            calendarId=calendar_id,
            body=event_body
        ).execute()

        return {
            "id": created_event.get('id'),
            "summary": created_event.get('summary'),
            "html_link": created_event.get('htmlLink'),
            "start": created_event.get('start', {}).get('dateTime'),
            "end": created_event.get('end', {}).get('dateTime'),
            "status": created_event.get('status')
        }

    def update_event(
        self,
        event_id: str,
        start_time: Any,
        end_time: Any,
        calendar_id: str = 'primary',
    ) -> Dict[str, Any]:
        """Updates the scheduled time of an existing clinic calendar event."""
        event = self.get_service().events().patch(
            calendarId=calendar_id,
            eventId=event_id,
            body={
                "start": {"dateTime": self._to_rfc3339(start_time)},
                "end": {"dateTime": self._to_rfc3339(end_time)},
            },
        ).execute()
        return {
            "id": event.get("id"),
            "start": event.get("start", {}).get("dateTime"),
            "end": event.get("end", {}).get("dateTime"),
            "status": event.get("status"),
        }

    def delete_event(
        self,
        event_id: str,
        calendar_id: str = 'primary',
    ) -> None:
        """Deletes a clinic calendar event by its stored event ID."""
        self.get_service().events().delete(
            calendarId=calendar_id,
            eventId=event_id,
        ).execute()

    def list_upcoming_events(
        self,
        max_results: int = 10,
        calendar_id: str = 'primary'
    ) -> List[Dict[str, Any]]:
        """Returns the next upcoming events starting from now."""
        now_rfc = datetime.now(timezone.utc).isoformat()
        service = self.get_service()

        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=now_rfc,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = []
        for item in events_result.get('items', []):
            events.append({
                "id": item.get('id'),
                "summary": item.get('summary', '(No title)'),
                "start": item.get('start', {}).get('dateTime', item.get('start', {}).get('date')),
                "end": item.get('end', {}).get('dateTime', item.get('end', {}).get('date')),
            })
        return events


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("=== Google Calendar Service Helper ===")
    cal = GoogleCalendarService()
    print(f"Client Secret: {cal.credentials_file}")
    print(f"Token File: {cal.token_file}")
    print(f"Is Authenticated: {cal.is_authenticated()}")

    if not cal.is_authenticated():
        print("\nStarting interactive OAuth authorization...")
        try:
            cal.authenticate_interactive()
            print("Authentication successful!")
        except Exception as err:
            print(f"Authentication failed: {err}")
    else:
        print("\nListing next 5 events...")
        try:
            events = cal.list_upcoming_events(max_results=5)
            print(f"Found {len(events)} events:")
            for ev in events:
                print(f" - {ev['summary']} ({ev['start']} to {ev['end']})")
        except Exception as err:
            print(f"Error fetching events: {err}")
