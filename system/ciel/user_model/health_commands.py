"""
Health Logging
------------------
The 'health' memory category has existed in MemoryStore's schema since
the very first session (Foundation Spec calls health a first-class
component, "sovereignty infrastructure") — and has never once been
written to. This closes that gap the same direct way as the user-model
commands: explicit statements from the person, logged with a real
timestamp, not inferred from anything.
"""

from __future__ import annotations

import datetime

from ciel.memory.store import MemoryStore


def log_health(entry: str) -> str:
    if not entry.strip():
        return "No health entry provided."

    memory = MemoryStore()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    content = f"[{timestamp}] {entry.strip()}"
    memory.add(category="health", content=content, source="user_reported", confidence=1.0)
    return f"Logged: {content}"


def health_summary(limit: int = 20) -> str:
    """Chronological view of everything logged — CIEL is NOT a medical
    professional and this makes no health claims or interpretations
    (per the Foundation Spec's explicit statement: 'CIEL is not a
    replacement for medical professionals'), it just shows what's been
    recorded, in order."""
    memory = MemoryStore()
    entries = memory.query(category="health", limit=limit)
    if not entries:
        return "No health entries logged yet. Use '!health <entry>' to start logging."

    lines = [f"Health log ({len(entries)} entries, most recent first):"]
    for e in entries:
        lines.append(f"  {e.content}")
    lines.append("\n(This is a record of what you've logged, not medical analysis or advice.)")
    return "\n".join(lines)
