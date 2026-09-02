"""
Sync Google Drive file metadata into CIEL's memory
=========================================================
Same pattern as sync_calendar_to_memory.py and sync_gmail_to_memory.py.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ciel.memory.store import MemoryStore
from google_drive import get_drive_service, get_recent_files

SOURCE_TAG = "google_drive"


def sync(max_results=15):
    print("Connecting to Google Drive...")
    service = get_drive_service()
    files = get_recent_files(service, max_results=max_results)

    memory = MemoryStore()
    removed = memory.delete_by_source(SOURCE_TAG)
    print(f"Cleared {removed} stale Drive memories from previous sync.")

    added = 0
    for f in files:
        content = f"Drive file: '{f['name']}' ({f['mimeType']}), last modified {f['modifiedTime']}"
        memory.add(category="user", content=content, source=SOURCE_TAG, confidence=1.0)
        added += 1

    print(f"Synced {added} recent files (metadata only, not content) into CIEL's memory.")
    return added


if __name__ == "__main__":
    sync()
