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
