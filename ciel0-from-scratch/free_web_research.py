"""
Free Web Research Tool — no Anthropic API, no payment, no dependency
=========================================================================

Fetches live internet data using only FREE, KEYLESS public APIs:
  1. DuckDuckGo Instant Answer API — good for direct factual queries
     ("what is X", definitions, well-known entities)
  2. Wikipedia's public search + summary API — fallback for broader queries

Neither requires an API key, a paid account, or any relationship with
Anthropic at all. This is genuinely independent: the only dependency is
your own internet connection.

This becomes the real handler for the 'web_research' tool registered in
tools/framework.py (which has sat unwired since Phase I). Run this file
directly to test it standalone, with no orchestrator, no Claude, no
billing involved whatsoever — just this script talking to the internet.
"""

import json
import urllib.request
import urllib.parse
from typing import Optional


def _http_get_json(url: str, timeout: int = 10) -> dict:
    """Plain HTTP GET using only Python's standard library — no third-party
    HTTP client dependency needed for this."""
    req = urllib.request.Request(url, headers={"User-Agent": "CIEL-research-tool/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def duckduckgo_instant_answer(query: str) -> Optional[str]:
    """DuckDuckGo's free Instant Answer API. No key required. Best for
    direct factual queries and well-known topics — NOT a general web
    search engine, more like an infobox lookup."""
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": 1, "skip_disambig": 1,
    })
    data = _http_get_json(url)
    abstract = data.get("AbstractText", "").strip()
    if abstract:
        source = data.get("AbstractSource", "DuckDuckGo")
        return f"{abstract} (source: {source})"
    return None


def wikipedia_summary(query: str) -> Optional[str]:
    """Fallback: search Wikipedia for the query, then fetch a summary of
    the top result. Two free, keyless calls: the search API to find the
    right page title, then the summary API to get real content."""
    search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query,
        "format": "json", "srlimit": 1,
    })
    search_data = _http_get_json(search_url)
    results = search_data.get("query", {}).get("search", [])
    if not results:
        return None
    title = results[0]["title"]

    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    summary_data = _http_get_json(summary_url)
    extract = summary_data.get("extract", "").strip()
    if extract:
        return f"{extract} (source: Wikipedia — {title})"
    return None


def web_research(query: str) -> str:
    """The actual tool handler: try DuckDuckGo first (faster, more direct
    for factual queries), fall back to Wikipedia (broader coverage)."""
    try:
        result = duckduckgo_instant_answer(query)
        if result:
            return result
    except Exception as e:
        print(f"  [DuckDuckGo lookup failed: {e}]")

    try:
        result = wikipedia_summary(query)
        if result:
            return result
    except Exception as e:
        print(f"  [Wikipedia lookup failed: {e}]")

    return f"No results found for '{query}' from either free source."


if __name__ == "__main__":
    print("=" * 60)
    print("Free web research tool — standalone test, zero Anthropic dependency")
    print("=" * 60)
    print("This calls DuckDuckGo's and Wikipedia's free public APIs directly.")
    print("No API key. No billing. No relationship with Anthropic at all.\n")

    test_queries = [
        "Python programming language",
        "current president of France",
        "what is a transformer neural network",
    ]

    for q in test_queries:
        print(f"Query: {q!r}")
        result = web_research(q)
        print(f"Result: {result}\n")

    print("Now try your own:")
    while True:
        q = input("Query (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        if not q:
            continue
        print(web_research(q), "\n")