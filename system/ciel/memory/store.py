"""
CIEL Memory Architecture
-------------------------
Per Foundation Spec §10 and Development Plan Phase I §3.

Categories: user, objectives, projects, knowledge, decisions,
experiences, lessons, skills, health.

Every memory record carries: creation date, source, confidence,
relevance, last verification, and optional expiration — CIEL must be
able to distinguish facts from assumptions.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sqlite3
from dataclasses import dataclass, asdict
from typing import Optional

CATEGORIES = (
    "user", "objectives", "projects", "knowledge",
    "decisions", "experiences", "lessons", "skills", "health",
)

_DB_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "ciel_data" / "memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.7,   -- 0.0-1.0; distinguishes fact from assumption
    relevance REAL NOT NULL DEFAULT 0.5,    -- decays / re-scored over time
    created_at TEXT NOT NULL,
    last_verified_at TEXT,
    expires_at TEXT,
    tags TEXT                                -- JSON list
);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
"""


@dataclass
class Memory:
    id: Optional[int]
    category: str
    content: str
    source: str
    confidence: float
    relevance: float
    created_at: str
    last_verified_at: Optional[str]
    expires_at: Optional[str]
    tags: list


class MemoryStore:
    def __init__(self, db_path: pathlib.Path = _DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def add(self, category: str, content: str, source: str,
            confidence: float = 0.7, relevance: float = 0.5,
            expires_at: Optional[str] = None, tags: Optional[list] = None) -> int:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown memory category '{category}'. Must be one of {CATEGORIES}")
        now = dt.datetime.utcnow().isoformat()
        cur = self.conn.execute(
            "INSERT INTO memories (category, content, source, confidence, relevance, "
            "created_at, last_verified_at, expires_at, tags) VALUES (?,?,?,?,?,?,?,?,?)",
            (category, content, source, confidence, relevance, now, now, expires_at,
             json.dumps(tags or [])),
        )
        self.conn.commit()
        return cur.lastrowid

    def query(self, category: Optional[str] = None, min_confidence: float = 0.0,
              limit: int = 20, text_contains: Optional[str] = None) -> list[Memory]:
        sql = "SELECT * FROM memories WHERE confidence >= ?"
        params: list = [min_confidence]
        if category:
            sql += " AND category = ?"
            params.append(category)
        if text_contains:
            sql += " AND content LIKE ?"
            params.append(f"%{text_contains}%")
        # exclude expired
        now = dt.datetime.utcnow().isoformat()
        sql += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(now)
        sql += " ORDER BY relevance DESC, created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [
            Memory(
                id=r["id"], category=r["category"], content=r["content"], source=r["source"],
                confidence=r["confidence"], relevance=r["relevance"], created_at=r["created_at"],
                last_verified_at=r["last_verified_at"], expires_at=r["expires_at"],
                tags=json.loads(r["tags"] or "[]"),
            )
            for r in rows
        ]

    def all_categories_summary(self) -> dict:
        cur = self.conn.execute("SELECT category, COUNT(*) c FROM memories GROUP BY category")
        return {row["category"]: row["c"] for row in cur.fetchall()}
