# CIEL — Phase I Scaffold

This is a working implementation of the Phase I architecture described in
`CIEL_Three_Phase_Development_Plan.docx`, built against
`CIEL_Foundation_Specification.docx`. It runs today, end to end, using
Claude as the interim reasoning engine in place of the from-scratch
CIEL-0 model.

## What's implemented

| Spec requirement | File | Status |
|---|---|---|
| Constitution (stored separately, not prompt content) | `ciel/config/constitution.yaml`, `ciel/constitution/loader.py` | Done — hash-verified, immutable at runtime |
| User Model | `ciel/user_model/profile.py` | Done — permanent/current/objectives/constraints/health |
| Memory Architecture | `ciel/memory/store.py` | Done — 9 categories, confidence/provenance/expiration |
| Skill Framework | `ciel/skills/registry.py` | Done — 10 seeded skills (foundational + master), versioned JSON |
| Tool Framework | `ciel/tools/framework.py` | Done — permission-controlled (allowed / requires_authorization / forbidden); most handlers still unwired |
| Cognitive Orchestrator | `ciel/orchestrator/core.py` | Done — User → Context/Memory/Skills/Tools → Reasoning → Response → Evaluation → Memory |
| Evaluation System | `ciel/evaluation/evaluator.py` | Done — logs the 10 named metrics per interaction |
| CIEL-0 (from-scratch model) | — | **Not started, deliberately** (see rationale below) |

## Why CIEL-0 wasn't the first thing built

Training a model from scratch without a working orchestrator/memory/skill
loop means training in a vacuum — there's no working system to validate
architectural decisions against, and no way to tell whether the model's
outputs are actually useful for CIEL's purpose. The Development Plan's own
Phase II §7 treats external models as legitimate "cognitive resources,"
so using Claude as the interim brain is inside spec, not a deviation from
it. The orchestrator calls it through one function
(`Orchestrator.process`) — when CIEL-0 exists, it's a drop-in replacement
behind that same interface. Nothing else in the architecture needs to
change.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python main.py            # interactive chat
python main.py --status   # inspect loaded constitution/skills/tools/memory
```

Without `ANTHROPIC_API_KEY` set, everything runs (constitution loads,
memory/skills/tools/evaluation all work) except the actual reasoning call.

## Data

Runtime state lives in `ciel_data/` (gitignored candidate — this is user
data, not code):
- `memory.db` — SQLite memory store
- `user_profile.json` — user model
- `skills/*.json` — one file per skill, independently versioned
- `evaluation_log.jsonl` — append-only evaluation history

## Immediate next steps (in priority order)

1. **Populate the real user profile** — the current state, constraints,
   objectives sections are empty placeholders. This should happen before
   much else, since the orchestrator's context assembly depends on it.
2. **Wire real tool handlers** — `web_research`, `code_execution`,
   `file_system` etc. are registered with correct permissions but have no
   handler yet (`Tool.invoke` raises `NotImplementedError`). Web research
   is the highest-leverage one to wire first.
3. **Replace placeholder evaluation scores** — `orchestrator/core.py`
   currently logs static scores per interaction; this needs either
   self-critique prompting or a separate grading pass.
4. **Memory relevance scoring** — retrieval is currently recency-ordered
   with no real relevance/embedding-based ranking. Fine at low memory
   volume; won't scale.
5. Only after 1–4 are stable and in daily use: revisit whether CIEL-0
   training is worth starting, per Phase I §4.
