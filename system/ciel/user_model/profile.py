"""
CIEL User Model
-----------------
Per Foundation Spec §9 and Development Plan Phase I §2.

Fields: permanent info, current state, objectives, desired future state,
history, constraints, health state (dedicated subsystem, not mixed into
general memory).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
from typing import Optional

_PROFILE_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "ciel_data" / "user_profile.json"


@dataclasses.dataclass
class HealthState:
    physical: dict = dataclasses.field(default_factory=dict)   # weight, fitness, sleep, nutrition...
    medical: dict = dataclasses.field(default_factory=dict)    # history, diagnoses, medications, allergies
    mental: dict = dataclasses.field(default_factory=dict)     # stress, mood, energy, focus, burnout
    lifestyle: dict = dataclasses.field(default_factory=dict)  # workload, recovery, exercise consistency


@dataclasses.dataclass
class UserProfile:
    permanent: dict = dataclasses.field(default_factory=dict)   # long-term traits, preferences, principles
    current_state: dict = dataclasses.field(default_factory=dict)
    objectives: list = dataclasses.field(default_factory=list)  # [{text, horizon, status, priority}]
    desired_future_state: dict = dataclasses.field(default_factory=dict)
    history: list = dataclasses.field(default_factory=list)     # [{date, decision, outcome, lesson}]
    constraints: dict = dataclasses.field(default_factory=dict) # financial, temporal, technical, legal, physical, environmental
    health: HealthState = dataclasses.field(default_factory=HealthState)
    last_updated: str = dataclasses.field(default_factory=lambda: dt.datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "UserProfile":
        health = HealthState(**d.get("health", {}))
        d = {**d, "health": health}
        return UserProfile(**d)

    def summary_for_prompt(self, max_objectives: int = 5) -> str:
        """Compact textual summary for injection into orchestrator context.
        Deliberately compact — full detail should be queried from memory,
        not dumped wholesale into every prompt."""
        parts = []
        if self.permanent:
            parts.append(f"Permanent traits/preferences: {json.dumps(self.permanent)}")
        if self.current_state:
            parts.append(f"Current state: {json.dumps(self.current_state)}")
        if self.objectives:
            top = self.objectives[:max_objectives]
            parts.append(f"Active objectives: {json.dumps(top)}")
        if self.constraints:
            parts.append(f"Constraints: {json.dumps(self.constraints)}")
        return "\n".join(parts) if parts else "No user profile data recorded yet."


class UserProfileStore:
    def __init__(self, path: pathlib.Path = _PROFILE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.save(UserProfile())

    def load(self) -> UserProfile:
        data = json.loads(self.path.read_text())
        return UserProfile.from_dict(data)

    def save(self, profile: UserProfile) -> None:
        profile.last_updated = dt.datetime.utcnow().isoformat()
        self.path.write_text(json.dumps(profile.to_dict(), indent=2))
