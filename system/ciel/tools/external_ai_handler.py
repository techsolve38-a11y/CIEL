"""
External AI Models Tool Handler
------------------------------------
Wires the 'external_ai_models' tool (registered since Phase I, never
implemented) to something real: CIEL-0 can delegate a specific task to
Ollama or Claude, reusing the SAME engine classes already built in
orchestrator/engines.py — no new integration code, just a new way of
using what already exists.

DELIBERATE DESIGN CHOICE: this is NOT auto-triggered by a keyword
heuristic the way Research or the calculator are. CIEL-0 cannot judge
its own output quality, so there's no reliable signal for "I'm stuck,
delegate this" — auto-triggering would just mean always delegating,
which defeats the entire "sub-task resource, not primary brain" idea
that was the explicit reason this tool was built. This stays an
explicit, deliberately-invoked escape hatch (see main.py's '!ask'
command), not something woven into every response.
"""

from __future__ import annotations


def call_external_model(query: str, model: str = "ollama", ollama_model: str = "llama3.2") -> str:
    """Delegates a single query to an external model, reusing the engine
    classes already built and tested. Defaults to Ollama (free, local)
    rather than Claude (paid), matching the project's stated priorities —
    Claude remains available by explicit choice (model='claude'), not as
    the default."""
    from ciel.orchestrator.engines import OllamaEngine, ClaudeEngine

    if not query.strip():
        return "No query provided to delegate."

    system_prompt = (
        "You are being consulted as a specialist resource by CIEL, a "
        "personal intelligence system. Answer the following directly and "
        "concisely — you are not CIEL itself, just a resource it's using "
        "for this one task."
    )

    if model == "ollama":
        engine = OllamaEngine(model=ollama_model)
        return engine.generate(system_prompt, query)
    elif model == "claude":
        import os
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "Cannot delegate to Claude: ANTHROPIC_API_KEY is not set."
        client = Anthropic(api_key=api_key)
        engine = ClaudeEngine(client=client, model="claude-sonnet-4-6", tools=[], memory_store=None)
        return engine.generate(system_prompt, query)
    else:
        return f"Unknown model '{model}'. Use 'ollama' or 'claude'."
