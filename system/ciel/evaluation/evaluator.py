"""
CIEL Evaluation System
------------------------
Per Development Plan Phase I §8. Tracks: accuracy, reasoning quality,
task completion, reliability, hallucination rate, tool-use accuracy,
memory accuracy, safety, user satisfaction, time/cost efficiency.

This is intentionally a lightweight, self-reported/log-based system at
Phase I — its job is to exist and accumulate a track record, not to be
a sophisticated auto-grader yet.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib

_LOG_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "ciel_data" / "evaluation_log.jsonl"

METRICS = (
    "accuracy", "reasoning_quality", "task_completion", "reliability",
    "hallucination_rate", "tool_use_accuracy", "memory_accuracy",
    "safety", "user_satisfaction", "time_cost_efficiency",
)


@dataclasses.dataclass
class EvaluationRecord:
    timestamp: str
    interaction_summary: str
    scores: dict            # subset of METRICS -> 0.0-1.0
    notes: str = ""


class Evaluator:
    def __init__(self, log_path: pathlib.Path = _LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, interaction_summary: str, scores: dict, notes: str = "") -> None:
        unknown = set(scores) - set(METRICS)
        if unknown:
            raise ValueError(f"Unknown evaluation metric(s): {unknown}")
        record = EvaluationRecord(
            timestamp=dt.datetime.utcnow().isoformat(),
            interaction_summary=interaction_summary,
            scores=scores,
            notes=notes,
        )
        with self.log_path.open("a") as f:
            f.write(json.dumps(dataclasses.asdict(record)) + "\n")

    def recent(self, n: int = 20) -> list[dict]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-n:]]

    def averages(self) -> dict:
        records = self.recent(n=10_000)
        sums: dict = {}
        counts: dict = {}
        for r in records:
            for k, v in r["scores"].items():
                sums[k] = sums.get(k, 0) + v
                counts[k] = counts.get(k, 0) + 1
        return {k: round(sums[k] / counts[k], 3) for k in sums}
