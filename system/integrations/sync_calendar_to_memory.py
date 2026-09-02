"""
Sync Google Calendar into CIEL's memory
=============================================

Fetches your upcoming events and writes them into the same MemoryStore
the orchestrator reads from — this is what turns "a script that prints
my calendar" into "CIEL actually knows my schedule when reasoning."

Run this periodically (manually for now; scheduling it is a later step)
to keep CIEL's calendar knowledge current. Each run REPLACES the
previous sync's events rather than piling up duplicates.

FOLDER LAYOUT this expects:
    system/
      ciel/                          <- the package main.py already uses
      integrations/
        google_calendar.py           <- move your existing google_calendar_sync.py here, rename it
        sync_calendar_to_memory.py   <- this file
        credentials.json             <- move here too
        token.json                   <- created automatically, stays here
"""

import sys
import pathlib

# Make the `ciel` package (one level up, in system/) importable regardless
# of which directory this script is actually run from.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ciel.memory.store import MemoryStore
from google_calendar import get_calendar_service, get_upcoming_events

SOURCE_TAG = "google_calendar"


def sync(max_results=20):
    print("Connecting to Google Calendar...")
    service = get_calendar_service()
    events = get_upcoming_events(service, max_results=max_results)

    memory = MemoryStore()
    removed = memory.delete_by_source(SOURCE_TAG)
    print(f"Cleared {removed} stale calendar memories from previous sync.")

    added = 0
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        summary = event.get("summary", "(no title)")
        content = f"Upcoming calendar event: '{summary}' at {start}"
        memory.add(
            category="user",
            content=content,
            source=SOURCE_TAG,
            confidence=1.0,   # this is real, verified data from your own calendar — not a guess
        )
        added += 1

    print(f"Synced {added} upcoming events into CIEL's memory.")
    return added


if __name__ == "__main__":
    sync()