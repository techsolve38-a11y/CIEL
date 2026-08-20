"""
CIEL Cognitive Orchestrator
------------------------------
Per Development Plan Phase I §5:

  Core flow: User -> CIEL Orchestrator -> Context/Memory/Skills/Tools
             -> Reasoning -> Action/Response -> Evaluation -> Memory.

At Phase I, the "reasoning" step is delegated to an external LLM
(Claude) acting as CIEL's interim cognitive engine — per Development
Plan Phase II §7, external models are cognitive resources CIEL uses;
they are not CIEL itself. The from-scratch CIEL-0 model can be dropped
in later behind this same interface without touching the surrounding
architecture (constitution, memory, skills, tools, evaluation).
"""

from __future__ import annotations

import os

from anthropic import Anthropic

from ciel.constitution.loader import load_constitution, verify_integrity
from ciel.memory.store import MemoryStore
from ciel.user_model.profile import UserProfileStore
from ciel.skills.registry import SkillRegistry, seed_default_skills
from ciel.tools.framework import default_tool_registry
from ciel.evaluation.evaluator import Evaluator

INTERIM_MODEL = "claude-sonnet-4-6"


class Orchestrator:
    def __init__(self):
        self.constitution = load_constitution()
        if not verify_integrity(self.constitution):
            raise RuntimeError(
                "CIEL constitution failed integrity verification. "
                "Refusing to start until this is resolved."
            )
        self.memory = MemoryStore()
        self.profile_store = UserProfileStore()
        self.skills = SkillRegistry()
        seed_default_skills(self.skills)
        self.tools = default_tool_registry()
        self.evaluator = Evaluator()

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._client = Anthropic(api_key=api_key) if api_key else None

    # ---- Context assembly -------------------------------------------------
    def _assemble_context(self, user_input: str) -> str:
        profile = self.profile_store.load()
        relevant_memories = self.memory.query(text_contains=None, limit=10)
        skill_summaries = self.skills.list_active_summaries()
        tool_summaries = self.tools.list_summaries()

        parts = [
            self.constitution.as_system_context(),
            "",
            "=== USER MODEL ===",
            profile.summary_for_prompt(),
            "",
            "=== RECENT MEMORY (most relevant) ===",
            "\n".join(f"- [{m.category}, confidence={m.confidence}] {m.content}" for m in relevant_memories) or "None yet.",
            "",
            "=== AVAILABLE SKILLS ===",
            "\n".join(f"- {s['name']} ({s['category']}): {s['purpose']}" for s in skill_summaries),
            "",
            "=== AVAILABLE TOOLS (this Phase I build; most are not yet wired to real handlers) ===",
            "\n".join(f"- {t['name']} [{t['permission']}]: {t['description']}" for t in tool_summaries),
            "",
            "Respond as CIEL. Be direct, honest about uncertainty, and grounded "
            "in the constitution above rather than generic assistant behavior.",
        ]
        return "\n".join(parts)

    # ---- Core loop ----------------------------------------------------------
    def process(self, user_input: str) -> str:
        if self._client is None:
            return (
                "[CIEL orchestrator is wired and ready, but no ANTHROPIC_API_KEY is set "
                "in the environment, so the reasoning engine can't be called yet. "
                "Set ANTHROPIC_API_KEY to activate reasoning.]"
            )

        system_context = self._assemble_context(user_input)
        response = self._client.messages.create(
            model=INTERIM_MODEL,
            max_tokens=1500,
            system=system_context,
            messages=[{"role": "user", "content": user_input}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")

        # Minimal evaluation logging (Phase I: self-reported placeholder scores
        # until a real grading mechanism exists — presence of the record matters
        # more than its precision at this stage).
        self.evaluator.log(
            interaction_summary=user_input[:120],
            scores={"task_completion": 1.0, "safety": 1.0},
        )

        # Store the exchange itself as an experience memory.
        self.memory.add(
            category="experiences",
            content=f"User asked: {user_input[:200]} | CIEL responded: {text[:200]}",
            source="orchestrator",
            confidence=1.0,
        )
        return text
