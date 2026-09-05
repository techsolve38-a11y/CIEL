"""
User Model Management
--------------------------
The UserProfile (permanent traits, objectives, constraints) has existed
since the very first session but stayed empty — everything built since
then reads from it, but nothing ever wrote to it except manually, once,
as a test. This gives a direct way to actually populate it: explicit
statements from the person, not inferred or guessed at.
"""

from __future__ import annotations

from ciel.user_model.profile import UserProfileStore


def remember_fact(fact: str) -> str:
    """Adds a permanent fact about the person — preferences, background,
    anything durable. Stored under a growing 'facts' list rather than
    trying to guess a more specific field, since we don't know the shape
    of what someone will want to record."""
    if not fact.strip():
        return "No fact provided to remember."

    store = UserProfileStore()
    profile = store.load()
    facts = profile.permanent.get("facts", [])
    facts.append(fact.strip())
    profile.permanent["facts"] = facts
    store.save(profile)
    return f"Remembered: '{fact.strip()}' (permanent facts now: {len(facts)})"


def add_objective(text: str, priority: int = 1, horizon: str = "unspecified") -> str:
    """Adds a structured objective — matches the shape UserProfile was
    always meant to hold (text/horizon/status/priority), not just a
    string, so it can be queried and prioritized later."""
    if not text.strip():
        return "No objective text provided."

    store = UserProfileStore()
    profile = store.load()
    objective = {"text": text.strip(), "horizon": horizon, "status": "active", "priority": priority}
    profile.objectives.append(objective)
    store.save(profile)
    return f"Objective added: '{text.strip()}' (priority {priority}, {len(profile.objectives)} total active objectives)"


def complete_objective(search_text: str) -> str:
    """Marks an objective as completed via partial, case-insensitive text
    match — objectives were addable but never completable until now,
    which meant the list could only ever grow."""
    if not search_text.strip():
        return "No objective text provided to match against."

    store = UserProfileStore()
    profile = store.load()
    search_lower = search_text.strip().lower()

    matches = [o for o in profile.objectives if o["status"] == "active" and search_lower in o["text"].lower()]

    if not matches:
        return f"No active objective found matching '{search_text}'."
    if len(matches) > 1:
        matched_texts = "; ".join(f"'{m['text']}'" for m in matches)
        return f"Multiple active objectives match '{search_text}' — be more specific: {matched_texts}"

    matches[0]["status"] = "completed"
    store.save(profile)
    return f"Marked as completed: '{matches[0]['text']}'"


def help_text() -> str:
    return """CIEL commands:
  !ask <question>       Delegate a question to Ollama (or Claude, if configured) as a scoped resource
  !remember <fact>       Save a permanent fact about yourself
  !objective <text>      Add an objective
  !complete <text>       Mark an objective as completed (partial text match)
  !health <entry>        Log a health-related entry
  !healthlog              Show your health log
  !whoami                 Show everything CIEL knows about you directly (facts + objectives)
  status                  Show system status (constitution, skills, tools, memory)
  exit / quit             End the session
Anything else is sent to CIEL-0 as an ordinary message."""


def list_user_model() -> str:
    """Shows what CIEL currently knows about the person directly (not
    from synced data — from what's actually been told to it)."""
    store = UserProfileStore()
    profile = store.load()
    lines = ["What CIEL knows about you (directly stated, not synced from Calendar/Gmail/Drive):"]

    facts = profile.permanent.get("facts", [])
    if facts:
        lines.append(f"\nFacts ({len(facts)}):")
        for f in facts:
            lines.append(f"  - {f}")
    else:
        lines.append("\nNo facts recorded yet.")

    if profile.objectives:
        lines.append(f"\nObjectives ({len(profile.objectives)}):")
        for o in profile.objectives:
            lines.append(f"  - [{o['status']}, priority {o['priority']}] {o['text']}")
    else:
        lines.append("\nNo objectives recorded yet.")

    return "\n".join(lines)