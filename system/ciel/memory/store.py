from __future__ import annotations
import datetime as dt, json, pathlib, sqlite3
from dataclasses import dataclass
from typing import Optional

CATEGORIES = ("user","objectives","projects","knowledge","decisions","experiences","lessons","skills","health")
_DB_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "ciel_data" / "memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL, content TEXT NOT NULL,
    source TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 0.7, relevance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL, last_verified_at TEXT, expires_at TEXT, tags TEXT
);
"""

@dataclass
class Memory:
    id: Optional[int]; category: str; content: str; source: str
    confidence: float; relevance: float; created_at: str
    last_verified_at: Optional[str]; expires_at: Optional[str]; tags: list

class MemoryStore:
    def __init__(self, db_path=_DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: needed because FastAPI (and other web
        # servers) run synchronous request handlers in a thread pool, so
        # a single MemoryStore instance created once at server startup
        # will legitimately be called from different threads across
        # different requests. SQLite's default blocks that as an extra
        # safety check. For this use case (a single local server, one
        # user, no truly simultaneous writes), disabling that check is
        # the standard, safe fix. It would need a more careful approach
        # (e.g. a connection per request, or a proper thread-safe pool)
        # if this were ever serving many concurrent users at once.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def add(self, category, content, source, confidence=0.7, relevance=0.5, expires_at=None, tags=None):
        now = dt.datetime.utcnow().isoformat()
        cur = self.conn.execute(
            "INSERT INTO memories (category,content,source,confidence,relevance,created_at,last_verified_at,expires_at,tags) VALUES (?,?,?,?,?,?,?,?,?)",
            (category, content, source, confidence, relevance, now, now, expires_at, json.dumps(tags or [])))
        self.conn.commit()
        return cur.lastrowid

    def query(self, category=None, min_confidence=0.0, limit=20, text_contains=None):
        sql = "SELECT * FROM memories WHERE confidence >= ?"
        params = [min_confidence]
        if category:
            sql += " AND category = ?"; params.append(category)
        now = dt.datetime.utcnow().isoformat()
        sql += " AND (expires_at IS NULL OR expires_at > ?) ORDER BY relevance DESC, created_at DESC LIMIT ?"
        params += [now, limit]
        rows = self.conn.execute(sql, params).fetchall()
        return [Memory(id=r["id"],category=r["category"],content=r["content"],source=r["source"],
                confidence=r["confidence"],relevance=r["relevance"],created_at=r["created_at"],
                last_verified_at=r["last_verified_at"],expires_at=r["expires_at"],tags=json.loads(r["tags"] or "[]")) for r in rows]

    def all_categories_summary(self):
        cur = self.conn.execute("SELECT category, COUNT(*) c FROM memories GROUP BY category")
        return {row["category"]: row["c"] for row in cur.fetchall()}

    def delete_by_source(self, source):
        """Remove all memories from a given source before re-syncing —
        without this, every calendar sync would ADD another copy of every
        event rather than refreshing them, and memory would fill with
        stale duplicates."""
        cur = self.conn.execute("DELETE FROM memories WHERE source = ?", (source,))
        self.conn.commit()
        return cur.rowcount