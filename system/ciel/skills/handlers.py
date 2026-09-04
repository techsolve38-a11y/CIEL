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


def research_handler(tool_input: dict, memory_store) -> str:
    """Real, free web research — DuckDuckGo + Wikipedia, no API key, no
    cost. Works identically regardless of which reasoning engine is
    running, since it's plain Python making HTTP calls, not something
    routed through any LLM provider."""
    from ciel.skills.free_research import web_research
    query = tool_input.get("query", "").strip()
    if not query:
        return "No query provided to research."
    return web_research(query)


# Keywords suggesting the user wants current/factual information CIEL
# should look up rather than guess — used ONLY for engines without native
# tool-use (ciel0, ollama), where Claude's "decide when to call a tool"
# mechanism isn't available. This is a deliberately simple heuristic, not
# a smart classifier — it will have false positives and misses, and
# that's an accepted, honest trade-off for reliability across any model.
RESEARCH_TRIGGER_KEYWORDS = (
    "what is", "who is", "when did", "when was", "current", "latest",
    "look up", "search for", "how many", "where is",
)


def should_trigger_research(user_input: str) -> bool:
    lowered = user_input.lower()
    return any(kw in lowered for kw in RESEARCH_TRIGGER_KEYWORDS)


def activity_summary_handler(tool_input: dict, memory_store) -> str:
    """Cross-references ALL connected data sources (calendar, gmail, drive,
    past experiences) into one summary. Distinct from pattern_recognition
    (which looks for RECURRENCE) — this one is about breadth: what's
    actually going on right now, across everything CIEL has access to."""
    all_memories = memory_store.query(limit=200)
    if not all_memories:
        return "No data available to summarize yet."

    by_source: dict[str, list] = {}
    for m in all_memories:
        by_source.setdefault(m.source, []).append(m.content)

    lines = [f"Activity summary across {len(by_source)} connected sources:"]
    source_labels = {
        "google_calendar": "Upcoming calendar events",
        "gmail": "Recent emails",
        "google_drive": "Recently modified files",
        "orchestrator": "Recent CIEL conversations",
    }
    for source, items in by_source.items():
        label = source_labels.get(source, source)
        lines.append(f"\n{label} ({len(items)}):")
        for item in items[:5]:   # cap per-source to keep the summary readable
            lines.append(f"  - {item}")
        if len(items) > 5:
            lines.append(f"  ... and {len(items) - 5} more")

    return "\n".join(lines)


ACTIVITY_SUMMARY_TRIGGER_KEYWORDS = (
    "summarize", "summary", "recap", "catch me up", "what have i been",
    "what's going on", "whats going on", "overview", "update me",
)


def should_trigger_activity_summary(user_input: str) -> bool:
    lowered = user_input.lower()
    return any(kw in lowered for kw in ACTIVITY_SUMMARY_TRIGGER_KEYWORDS)


import re


def calculator_skill_handler(tool_input: dict, memory_store) -> str:
    from ciel.tools.calculator_handler import calculate
    expression = tool_input.get("expression", "").strip()
    if not expression:
        return "No expression provided to calculate."
    return calculate(expression)


# Matches a contiguous arithmetic expression: starts and ends with a
# digit, with only digits/operators/parens/decimal points/spaces in
# between — this naturally excludes surrounding English words, since
# letters aren't in the allowed character set.
_MATH_LIKE_PATTERN = re.compile(r"\d[\d\.\+\-\*/%\(\)\s]*\d")


def extract_math_expression(user_input: str):
    match = _MATH_LIKE_PATTERN.search(user_input)
    if not match:
        return None
    candidate = match.group().strip()
    # Require at least one actual operator — otherwise a bare number
    # sequence (e.g. part of a date or phone number) would incorrectly
    # look like something to calculate.
    if not any(op in candidate for op in "+-*/%"):
        return None
    # Reject tightly-packed digit-hyphen sequences like "2026-08-30" —
    # real arithmetic typed in natural language almost always has spaces
    # around operators ("23 * 47"); dates and similar identifiers don't.
    # A simple, honest heuristic, not a perfect classifier.
    if " " not in candidate:
        return None
    return candidate


import datetime


def _parse_event_datetime(content: str):
    """Extract and parse a date/datetime from calendar memory content like
    \"Upcoming calendar event: 'X' at 2026-08-30T09:00:00\". Returns None
    (not an exception) for anything unparseable — a malformed or
    unexpected date format should be silently skipped, not crash the
    whole urgency scan over one bad entry."""
    match = re.search(r"at (\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?)", content)
    if not match:
        return None
    date_str = match.group(1)
    try:
        if "T" in date_str:
            return datetime.datetime.fromisoformat(date_str)
        # All-day events (Drive/Calendar sometimes give date-only, no time)
        return datetime.datetime.fromisoformat(date_str + "T00:00:00")
    except ValueError:
        return None


def urgency_handler(tool_input: dict, memory_store) -> str:
    """Scans calendar-sourced memory for items happening soon — real date
    math against the actual current time, not a guess."""
    hours_threshold = tool_input.get("hours_threshold", 72)
    memories = memory_store.query(category="user", limit=200)
    now = datetime.datetime.now()

    urgent, upcoming, unparseable = [], [], 0
    for m in memories:
        if m.source != "google_calendar":
            continue
        event_time = _parse_event_datetime(m.content)
        if event_time is None:
            unparseable += 1
            continue
        hours_until = (event_time - now).total_seconds() / 3600
        if hours_until < 0:
            continue   # already past — not urgent, just old
        if hours_until <= hours_threshold:
            urgent.append((hours_until, m.content))
        else:
            upcoming.append((hours_until, m.content))

    lines = [f"Urgency scan (threshold: {hours_threshold} hours from now):"]
    if urgent:
        lines.append(f"\nURGENT — within {hours_threshold} hours ({len(urgent)}):")
        for hours, content in sorted(urgent):
            lines.append(f"  - [{hours:.1f}h away] {content}")
    else:
        lines.append(f"\nNothing urgent within {hours_threshold} hours.")
    if upcoming:
        lines.append(f"\nFurther out ({len(upcoming)}): " +
                     ", ".join(c for _, c in sorted(upcoming)[:3]) +
                     (f" ... and {len(upcoming)-3} more" if len(upcoming) > 3 else ""))
    if unparseable:
        lines.append(f"\n({unparseable} calendar entries had unparseable dates and were skipped.)")

    return "\n".join(lines)


URGENCY_TRIGGER_KEYWORDS = ("urgent", "deadline", "coming up", "soon", "today", "tomorrow", "this week")


def should_trigger_urgency(user_input: str) -> bool:
    lowered = user_input.lower()
    return any(kw in lowered for kw in URGENCY_TRIGGER_KEYWORDS)


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
    "Research": {
        "tool_schema": {
            "name": "research",
            "description": (
                "Look up current, factual, or general-knowledge information "
                "using free public sources (DuckDuckGo, Wikipedia). Use this "
                "when the user asks something you shouldn't guess at — facts, "
                "definitions, current events, or anything time-sensitive."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — as specific as possible.",
                    }
                },
                "required": ["query"],
            },
        },
        "handler": research_handler,
    },
    "Communication": {
        "tool_schema": {
            "name": "activity_summary",
            "description": (
                "Summarize recent activity across ALL connected personal data "
                "sources (calendar, email, files, past conversations). Use this "
                "when the user asks for a recap, overview, or 'catch me up.'"
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        "handler": activity_summary_handler,
    },
    "Capital Allocation": {
        "tool_schema": {
            "name": "calculate",
            "description": (
                "Evaluate a safe arithmetic expression (numbers and "
                "+, -, *, /, **, % operators only — no code execution). "
                "Use this for any math rather than guessing at arithmetic."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The arithmetic expression, e.g. '23 * 47 + 12'."}
                },
                "required": ["expression"],
            },
        },
        "handler": calculator_skill_handler,
    },
    "Reasoning": {
        "tool_schema": {
            "name": "urgency_scan",
            "description": (
                "Scan calendar data for events happening soon, flagging what's "
                "genuinely urgent based on real date comparison. Use this when "
                "the user asks what's urgent, coming up, due soon, or needs "
                "attention — this gives a real, computed answer, not a guess."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "hours_threshold": {"type": "number", "description": "Hours from now to count as urgent. Default 72."}
                },
            },
        },
        "handler": urgency_handler,
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