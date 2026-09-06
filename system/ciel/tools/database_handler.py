"""
Database Tool Handler
-------------------------
Gives the 'database' tool (registered since Phase I, previously
unwired) a real implementation: targeted search over CIEL's own memory,
rather than the "dump everything" approach pattern_recognition and
activity_summary use.
"""

from __future__ import annotations

from ciel.memory.store import MemoryStore


def query_memory(keyword: str = "", category: str = "") -> str:
    memory = MemoryStore()
    results = memory.query(category=category or None, limit=200)
    if keyword:
        results = [r for r in results if keyword.lower() in r.content.lower()]

    if not results:
        return f"No memory entries found matching keyword={keyword!r}, category={category!r}."

    lines = [f"Found {len(results)} matching entries:"]
    for r in results[:15]:
        lines.append(f"  [{r.source}] {r.content}")
    if len(results) > 15:
        lines.append(f"  ... and {len(results) - 15} more")
    return "\n".join(lines)