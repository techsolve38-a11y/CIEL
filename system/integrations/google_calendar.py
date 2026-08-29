"""
Google Calendar Sync — connecting CIEL to your real data
==============================================================

FIXED: credentials.json and token.json paths are now resolved relative
to THIS SCRIPT'S OWN LOCATION, not the current working directory. The
original version used bare relative paths ("credentials.json"), which
meant the script's behavior silently depended on which folder you
happened to run `python` from — placing the files in the "right" folder
still failed if you invoked the script from a different directory. This
version works correctly regardless of where you run it from, as long as
credentials.json/token.json sit next to this file.
"""

import datetime
import pathlib

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# Resolve relative to THIS FILE's directory, not the current working
# directory — this is the fix. __file__ is always this script's own
# path, so .parent is always the folder it lives in, regardless of
# where `python ...` was invoked from.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CREDENTIALS_PATH = _SCRIPT_DIR / "credentials.json"
TOKEN_PATH = _SCRIPT_DIR / "token.json"


def get_calendar_service():
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS_PATH}. "
                    f"Download it from Google Cloud Console and place it in "
                    f"{_SCRIPT_DIR} (the same folder as this script)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_PATH.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def get_upcoming_events(service, max_results=10):
    now = datetime.datetime.utcnow().isoformat() + "Z"
    events_result = service.events().list(
        calendarId="primary", timeMin=now, maxResults=max_results,
        singleEvents=True, orderBy="startTime",
    ).execute()
    return events_result.get("items", [])


if __name__ == "__main__":
    print("=" * 60)
    print("Google Calendar sync — connecting to YOUR real data")
    print("=" * 60)
    print(f"Looking for credentials.json / token.json in: {_SCRIPT_DIR}\n")

    service = get_calendar_service()
    events = get_upcoming_events(service, max_results=10)

    if not events:
        print("No upcoming events found.")
    else:
        print(f"Found {len(events)} upcoming events:\n")
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            summary = event.get("summary", "(no title)")
            print(f"  {start} — {summary}")