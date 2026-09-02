"""
Executable Skill Handlers
----------------------------
Per Foundation Spec, skills should be genuinely invokable capabilities,
not just descriptive text fed into the reasoning engine's context. This
module is the bridge: each entry here pairs a skill NAME with (a) an
Anthropic tool schema describing its inputs, and (b) a real Python
function that does the actual work when Claude decides to call it.

This is deliberately kept separate from skills/registry.py — that module
persists skill METADATA as JSON (purpose, version, performance history),
which can't hold a live Python function. This module holds the actual
callable logic, matched up by skill name at runtime.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable


def pattern_recognition_handler(tool_input: dict, memory_store) -> str:
    """Analyzes stored memory for recurring patterns — e.g. calendar events
    that repeat, suggesting a recurring commitment. Pure computation, no
    LLM call, so this costs nothing and runs instantly."""
    category = tool_input.get("category")
    memories = memory_store.query(category=category, limit=200) if category else memory_store.query(limit=200)

    if not memories:
        return "No memory entries found to analyze."

    # Extract quoted titles from content like "Upcoming calendar event: 'Team standup' at ..."
    titles = []
    for m in memories:
        match = re.search(r"'([^']+)'", m.content)
        if match:
            titles.append(match.group(1))

    counts = Counter(titles)
    recurring = {title: count for title, count in counts.items() if count > 1}

    lines = [f"Analyzed {len(memories)} memory entries across {len(set(m.category for m in memories))} categories."]
    if recurring:
        lines.append("Recurring patterns detected:")
        for title, count in sorted(recurring.items(), key=lambda x: -x[1]):
            lines.append(f"  - '{title}' appears {count} times — likely a recurring commitment.")
    else:
        lines.append("No recurring patterns detected (each item appears only once, or too little data yet).")

    category_counts = Counter(m.category for m in memories)
    lines.append("Memory distribution by category: " +
                 ", ".join(f"{cat}={n}" for cat, n in category_counts.most_common()))

    return "\n".join(lines)


# --- Registry: maps skill name -> (Anthropic tool schema, handler function) ---
# Only skills with a REAL implementation appear here. A skill can exist in
# skills/registry.py's metadata without appearing here — that just means
# it's still descriptive-only, not yet executable. This registry is the
# honest, growable record of which skills are actually real.
EXECUTABLE_SKILLS: dict[str, dict] = {
    "Pattern Recognition": {
        "tool_schema": {
            "name": "pattern_recognition",
            "description": (
                "Analyze CIEL's stored memory for recurring patterns — e.g. "
                "repeated calendar events suggesting a recurring commitment. "
                "Use this when the user asks about patterns, habits, recurring "
                "events, or trends in their own data, rather than guessing."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional: restrict analysis to one memory category "
                                        "(e.g. 'user', 'experiences'). Omit to analyze everything.",
                    }
                },
            },
        },
        "handler": pattern_recognition_handler,
    },
}


def get_executable_skill_tools() -> list[dict]:
    """Returns Anthropic tool definitions for every skill that actually has
    a real implementation — this is what gets added to the API call's
    `tools` list, alongside web_search."""
    return [entry["tool_schema"] for entry in EXECUTABLE_SKILLS.values()]


def execute_skill_by_tool_name(tool_name: str, tool_input: dict, memory_store):
    """Dispatches a tool_use block (identified by its tool NAME, which
    matches tool_schema['name'] above, not the skill's display name) to
    the correct handler."""
    for skill_name, entry in EXECUTABLE_SKILLS.items():
        if entry["tool_schema"]["name"] == tool_name:
            return entry["handler"](tool_input, memory_store)
    raise ValueError(f"No executable skill registered for tool name '{tool_name}'")