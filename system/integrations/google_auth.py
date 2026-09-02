"""
Shared Google OAuth
-----------------------
One authorization flow, covering Calendar + Gmail + Drive together —
refactored out of google_calendar.py, which previously had its own copy
of this logic. Adding Gmail and Drive means expanding the scopes, which
requires ONE re-consent (delete token.json, log in again) — after that,
this single token covers all three services.

Read-only, least-privilege scopes throughout, matching the same
reasoning as the original Calendar setup: CIEL should be able to SEE
your data, not modify or delete it, unless that's a deliberate, separate
decision later.
"""

import pathlib

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",  # file names/dates, not content —
                                                                    # a deliberate lower-privilege
                                                                    # starting point than full drive.readonly
]

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CREDENTIALS_PATH = _SCRIPT_DIR / "credentials.json"
TOKEN_PATH = _SCRIPT_DIR / "token.json"


def get_credentials():
    """Handles the OAuth flow for ALL configured scopes at once. Reuses a
    saved token if valid, refreshes if expired, or triggers a fresh
    browser login if neither exists yet (or if scopes changed since the
    last login — Google requires re-consent when scopes expand)."""
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid or set(creds.scopes or []) != set(SCOPES):
        if creds and creds.expired and creds.refresh_token and set(creds.scopes or []) == set(SCOPES):
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS_PATH}. "
                    f"Download it from Google Cloud Console and place it in {_SCRIPT_DIR}."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_PATH.write_text(creds.to_json())

    return creds


def get_service(name: str, version: str):
    """Build any Google API service (calendar/v3, gmail/v1, drive/v3, etc.)
    using the shared, already-authorized credentials."""
    creds = get_credentials()
    return build(name, version, credentials=creds)
