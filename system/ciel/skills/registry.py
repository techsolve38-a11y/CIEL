"""
CIEL Skill Framework
----------------------
Per Foundation Spec §10-13 and Development Plan Phase I §6.

Skills are modular DATA objects, not hard-coded functions, so they can
be created, modified, versioned and evolved at runtime without touching
core code. Each skill record has: name, purpose, description, inputs,
outputs, dependencies, knowledge_requirements, procedures, evaluation
criteria, performance_history, version.

Categories: foundational, master, adaptive (per Foundation Spec §10).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
from typing import Optional

_SKILLS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "ciel_data" / "skills"


@dataclasses.dataclass
class SkillPerformanceRecord:
    timestamp: str
    outcome: str          # "success" | "failure" | "partial"
    notes: str = ""


@dataclasses.dataclass
class Skill:
    name: str
    category: str          # foundational | master | adaptive
    purpose: str
    description: str
    inputs: list = dataclasses.field(default_factory=list)
    outputs: list = dataclasses.field(default_factory=list)
    dependencies: list = dataclasses.field(default_factory=list)   # other skill names
    knowledge_requirements: list = dataclasses.field(default_factory=list)
    procedures: list = dataclasses.field(default_factory=list)     # ordered steps / prompt guidance
    evaluation_criteria: list = dataclasses.field(default_factory=list)
    performance_history: list = dataclasses.field(default_factory=list)  # list[SkillPerformanceRecord dicts]
    version: str = "0.1.0"
    active: bool = True

    def record_outcome(self, outcome: str, notes: str = ""):
        self.performance_history.append(dataclasses.asdict(
            SkillPerformanceRecord(timestamp=dt.datetime.utcnow().isoformat(), outcome=outcome, notes=notes)
        ))


class SkillRegistry:
    """Loads/saves skills as individual JSON files so each is independently
    versionable and diffable (git-friendly), matching the "skills must be
    versioned and reversible" requirement in Phase II."""

    def __init__(self, skills_dir: pathlib.Path = _SKILLS_DIR):
        self.dir = skills_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> pathlib.Path:
        safe = name.lower().replace(" ", "_")
        return self.dir / f"{safe}.json"

    def save(self, skill: Skill) -> None:
        self._path(skill.name).write_text(json.dumps(dataclasses.asdict(skill), indent=2))

    def load(self, name: str) -> Optional[Skill]:
        p = self._path(name)
        if not p.exists():
            return None
        return Skill(**json.loads(p.read_text()))

    def list_all(self) -> list[Skill]:
        return [Skill(**json.loads(p.read_text())) for p in sorted(self.dir.glob("*.json"))]

    def list_active_summaries(self) -> list[dict]:
        return [
            {"name": s.name, "category": s.category, "purpose": s.purpose, "version": s.version}
            for s in self.list_all() if s.active
        ]


def seed_default_skills(registry: SkillRegistry) -> None:
    """Populate the initial skill set named explicitly in the Foundation
    Spec §10 and Development Plan §6, so the system isn't starting from
    an empty registry."""
    defaults = [
        Skill(name="Reasoning", category="foundational", purpose="General multi-step reasoning",
              description="Break down problems and reason through them systematically.",
              procedures=["Decompose problem", "Consider alternatives", "Reach conclusion with stated confidence"]),
        Skill(name="Research", category="foundational", purpose="Gather and verify information",
              description="Find, cross-check and summarize information from available tools.",
              dependencies=["Reasoning"]),
        Skill(name="Strategic Planning", category="foundational", purpose="Turn objectives into actionable plans",
              description="Convert Objective -> Strategy -> Tasks -> Actions -> Results -> Lessons.",
              dependencies=["Reasoning"]),
        Skill(name="Communication", category="foundational", purpose="Clear communication with the user",
              description="Communicate findings, uncertainty and recommendations clearly."),
        Skill(name="Pattern Recognition", category="foundational", purpose="Detect trends and anomalies",
              description="Identify patterns across memory, health data, and objectives."),
        Skill(name="Reverse Engineering", category="master", purpose="Deconstruct systems and businesses",
              description="Technical and business reverse engineering per Foundation Spec §13.",
              dependencies=["Research", "Reasoning"]),
        Skill(name="Corporate Acquisitions", category="master", purpose="Support the acquisition lifecycle",
              description="Find -> Understand -> Value -> Investigate -> Structure -> Negotiate -> Acquire -> Integrate -> Optimize -> Compound.",
              dependencies=["Reverse Engineering", "Capital Allocation"]),
        Skill(name="Capital Allocation", category="master", purpose="Evaluate and allocate capital",
              description="Assess opportunities by return, risk, capital/time requirement, scalability, ownership, sovereignty contribution."),
        Skill(name="Technology Strategy", category="master", purpose="Guide technology decisions strategically",
              description="Evaluate technology choices for ownership, leverage, scalability, sovereignty."),
        Skill(name="Business Building", category="master", purpose="Build and grow businesses",
              description="End-to-end support for building productive, ownable business capacity."),
    ]
    for s in defaults:
        if registry.load(s.name) is None:
            registry.save(s)
