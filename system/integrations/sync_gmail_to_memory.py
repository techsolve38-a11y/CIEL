"""
Sync Gmail metadata into CIEL's memory
============================================
Same pattern as sync_calendar_to_memory.py: fetch real data, clear the
previous sync's entries, write fresh ones. Run periodically to keep
CIEL's picture of your inbox current.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ciel.memory.store import MemoryStore
from google_gmail import get_gmail_service, get_recent_messages

SOURCE_TAG = "gmail"


def sync(max_results=15):
    print("Connecting to Gmail...")
    service = get_gmail_service()
    messages = get_recent_messages(service, max_results=max_results)

    memory = MemoryStore()
    removed = memory.delete_by_source(SOURCE_TAG)
    print(f"Cleared {removed} stale Gmail memories from previous sync.")

    added = 0
    for msg in messages:
        content = f"Recent email from {msg['from']}, subject: '{msg['subject']}' ({msg['date']})"
        memory.add(category="user", content=content, source=SOURCE_TAG, confidence=1.0)
        added += 1

    print(f"Synced {added} recent emails (metadata only, not full content) into CIEL's memory.")
    return added


if __name__ == "__main__":
    sync()
