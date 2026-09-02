"""
Google Calendar — now using the shared auth module (google_auth.py)
instead of its own OAuth logic. Behavior is identical to before; only
the authorization plumbing moved to be shared with Gmail and Drive.
"""

import datetime

from google_auth import get_service


def get_calendar_service():
    return get_service("calendar", "v3")


def get_upcoming_events(service, max_results=10):
    now = datetime.datetime.utcnow().isoformat() + "Z"
    events_result = service.events().list(
        calendarId="primary", timeMin=now, maxResults=max_results,
        singleEvents=True, orderBy="startTime",
    ).execute()
    return events_result.get("items", [])


if __name__ == "__main__":
    service = get_calendar_service()
    events = get_upcoming_events(service, max_results=10)
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        print(f"  {start} — {event.get('summary', '(no title)')}")
