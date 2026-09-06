"""
CIEL Cognitive Orchestrator
------------------------------
UPDATE (reasoning engine choice): per an explicit decision to prioritize
genuine independence over working polish, CIEL-0 is now the DEFAULT
reasoning engine — no external model, no API, no dependency on anyone
else's weights, paid or free. Ollama and Claude remain available as
options (Orchestrator(engine="ollama") / engine="claude") for later, but
neither is the default anymore.

HONEST STATE: CIEL-0 today cannot hold a real conversation — this was
proven directly in testing, not assumed. Choosing engine="ciel0" means
choosing a system that runs entirely independently, in exchange for
weak, often incoherent responses until CIEL-0 grows. That trade-off was
made deliberately and explicitly, not by accident.
"""

from __future__ import annotations

import os

from ciel.constitution.loader import load_constitution, verify_integrity
from ciel.memory.store import MemoryStore
from ciel.user_model.profile import UserProfileStore
from ciel.skills.registry import SkillRegistry, seed_default_skills
from ciel.skills.handlers import (
    EXECUTABLE_SKILLS, pattern_recognition_handler, research_handler,
    should_trigger_research, activity_summary_handler, should_trigger_activity_summary,
    calculator_skill_handler, extract_math_expression, urgency_handler, should_trigger_urgency,
)
from ciel.tools.framework import default_tool_registry
from ciel.evaluation.evaluator import Evaluator
from ciel.orchestrator.engines import OllamaEngine, ClaudeEngine, CIEL0Engine

CLAUDE_MODEL = "claude-sonnet-4-6"
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}


class Orchestrator:
    def __init__(self, engine: str = "ciel0", enable_web_search: bool = True,
                 ollama_model: str = "llama3.2", ciel0_checkpoint: str = "ciel0_checkpoint.pt"):
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
        self.engine_name = engine

        if engine == "ciel0":
            self.engine = CIEL0Engine(checkpoint_path=ciel0_checkpoint)
        elif engine == "ollama":
            self.engine = OllamaEngine(model=ollama_model)
        elif engine == "claude":
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            client = Anthropic(api_key=api_key) if api_key else None
            if client is None:
                raise RuntimeError("engine='claude' requires ANTHROPIC_API_KEY to be set.")
            tools = [e["tool_schema"] for e in EXECUTABLE_SKILLS.values()]
            if enable_web_search:
                tools.append(WEB_SEARCH_TOOL)
            self.engine = ClaudeEngine(client=client, model=CLAUDE_MODEL, tools=tools, memory_store=self.memory)
        else:
            raise ValueError(f"Unknown engine '{engine}'. Use 'ciel0', 'ollama', or 'claude'.")

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
        ]

        # Neither CIEL-0 nor Ollama support Claude's native tool-use, so
        # cheap/free skills get computed proactively and folded directly
        # into context — reliable regardless of which engine is running.
        if self.engine_name in ("ciel0", "ollama"):
            if "Pattern Recognition" in EXECUTABLE_SKILLS:
                pattern_result = pattern_recognition_handler({}, self.memory)
                parts += ["", "=== COMPUTED: PATTERN RECOGNITION ===", pattern_result]

            if "Research" in EXECUTABLE_SKILLS and should_trigger_research(user_input):
                research_result = research_handler({"query": user_input}, self.memory)
                parts += ["", "=== COMPUTED: WEB RESEARCH (free, real result) ===", research_result]

            if "Communication" in EXECUTABLE_SKILLS and should_trigger_activity_summary(user_input):
                summary_result = activity_summary_handler({}, self.memory)
                parts += ["", "=== COMPUTED: ACTIVITY SUMMARY ===", summary_result]

            if "Capital Allocation" in EXECUTABLE_SKILLS:
                math_expr = extract_math_expression(user_input)
                if math_expr:
                    calc_result = calculator_skill_handler({"expression": math_expr}, self.memory)
                    parts += ["", "=== COMPUTED: CALCULATOR (real result) ===", calc_result]

            if "Reasoning" in EXECUTABLE_SKILLS and should_trigger_urgency(user_input):
                urgency_result = urgency_handler({}, self.memory)
                parts += ["", "=== COMPUTED: URGENCY SCAN (real date comparison) ===", urgency_result]

        parts += [
            "",
            "=== AVAILABLE SKILLS ===",
            "\n".join(f"- {s['name']} ({s['category']}): {s['purpose']}" for s in skill_summaries),
            "",
            "=== AVAILABLE TOOLS ===",
            "\n".join(f"- {t['name']} [{t['permission']}]: {t['description']}" for t in tool_summaries),
            "",
            "Respond as CIEL. Be direct and grounded in the constitution above.",
        ]
        return "\n".join(parts)

    # ---- Core loop ----------------------------------------------------------
    def process(self, user_input: str) -> str:
        import time
        from ciel.evaluation.scoring import compute_task_completion, compute_time_efficiency

        system_context = self._assemble_context(user_input)

        start = time.time()
        text = self.engine.generate(system_context, user_input)
        elapsed = time.time() - start

        # REAL evaluation, replacing the old hardcoded {task_completion: 1.0,
        # safety: 1.0} that got logged unconditionally on every interaction.
        # Only metrics we can honestly compute get logged — the rest
        # (accuracy, reasoning_quality, etc.) genuinely need human feedback
        # or an LLM judge, and we don't fake numbers for those.
        scores = {
            "task_completion": compute_task_completion(text),
            "time_cost_efficiency": compute_time_efficiency(elapsed),
        }
        self.evaluator.log(
            interaction_summary=user_input[:120], scores=scores,
            notes=f"engine={self.engine_name}, elapsed={elapsed:.2f}s",
        )
        self.memory.add(
            category="experiences",
            content=f"User asked: {user_input[:200]} | CIEL responded: {text[:200]}",
            source="orchestrator",
            confidence=1.0,
        )
        return text