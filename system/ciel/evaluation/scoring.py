"""
Real Evaluation Logic
--------------------------
Replaces the hardcoded {task_completion: 1.0, safety: 1.0} that got
logged on EVERY interaction regardless of what actually happened — that
wasn't evaluation, it was a stub pretending to be one.

Two things are honestly computable without needing another LLM to judge
itself:
  1. task_completion — did the response contain an actual answer, or one
     of CIEL's OWN known failure messages (API errors, connection
     failures, etc.)? We defined these exact messages ourselves
     throughout this project, so checking for them is a real signal,
     not a guess.
  2. time_cost_efficiency — real wall-clock time the engine call took,
     converted to a 0-1 score via an explicit, documented formula.

Everything else in METRICS (accuracy, reasoning_quality,
hallucination_rate, memory_accuracy, user_satisfaction) genuinely
requires either human feedback or an LLM judge to assess honestly.
Rather than fabricate numbers for those, we simply don't log them —
an honest gap is better than a fake value with false precision.
"""

from __future__ import annotations

# Known failure message fragments — these are messages CIEL ITSELF
# generates (see orchestrator/engines.py) when something goes wrong.
# Checking for them is checking our own documented failure modes, not
# guessing at unknown ones.
_KNOWN_FAILURE_MARKERS = (
    "couldn't be reached",
    "no anthropic_api_key is set",
    "anthropic_api_key is not set",
    "insufficient credits",
    "is rate-limited",
    "couldn't reach the anthropic api",
    "returned an unexpected error",
    "used the maximum number of tool calls",
    "skill execution failed",
    "unknown model",
    "no query provided",
)


def compute_task_completion(response_text: str) -> float:
    """1.0 if the response looks like a real answer, 0.0 if it matches
    one of CIEL's own known failure patterns, or is empty."""
    if not response_text or not response_text.strip():
        return 0.0
    lowered = response_text.lower()
    if any(marker in lowered for marker in _KNOWN_FAILURE_MARKERS):
        return 0.0
    return 1.0


def compute_time_efficiency(elapsed_seconds: float) -> float:
    """A real, measured wall-clock time, converted to a 0-1 score via an
    explicit formula: full credit under 2 seconds, linearly decreasing
    to 0 at 30 seconds, floored at 0 beyond that. The threshold values
    are a judgment call (documented, not hidden) — not a precise
    scientific measurement, but a genuine computation from a real
    timing, unlike the previous hardcoded 1.0."""
    if elapsed_seconds <= 2.0:
        return 1.0
    if elapsed_seconds >= 30.0:
        return 0.0
    return round(1.0 - (elapsed_seconds - 2.0) / 28.0, 3)
