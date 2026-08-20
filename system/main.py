#!/usr/bin/env python3
"""
CIEL — CLI entry point (Phase I prototype)

Usage:
    python main.py            # interactive chat loop
    python main.py --status   # print system status and exit
"""

import sys

from ciel.orchestrator.core import Orchestrator


def print_status(orch: Orchestrator):
    print("=== CIEL SYSTEM STATUS ===")
    print(f"Constitution version: {orch.constitution.version} (integrity OK)")
    print(f"Core Laws loaded: {len(orch.constitution.core_laws)}")
    print(f"Memory categories in use: {orch.memory.all_categories_summary()}")
    print(f"Skills registered: {len(orch.skills.list_all())}")
    for s in orch.skills.list_active_summaries():
        print(f"  - [{s['category']}] {s['name']} v{s['version']}: {s['purpose']}")
    print(f"Tools registered: {len(orch.tools.list_all())}")
    for t in orch.tools.list_summaries():
        print(f"  - {t['name']} [{t['permission']}]")
    evals = orch.evaluator.averages()
    print(f"Evaluation averages: {evals if evals else 'no interactions logged yet'}")
    print(f"Reasoning engine active: {'yes' if orch._client else 'no (set ANTHROPIC_API_KEY)'}")


def main():
    orch = Orchestrator()

    if "--status" in sys.argv:
        print_status(orch)
        return

    print("CIEL Phase I prototype. Type 'exit' to quit, 'status' for system status.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCIEL: Ending session.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("CIEL: Ending session.")
            break
        if user_input.lower() == "status":
            print_status(orch)
            continue

        response = orch.process(user_input)
        print(f"\nCIEL: {response}\n")


if __name__ == "__main__":
    main()
