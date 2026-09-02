"""
Free web research — DuckDuckGo + Wikipedia, no API key, no cost.
Moved here from the standalone free_web_research.py script so it can be
reused as a genuine skill handler, not just run standalone.
"""

import json
import urllib.request
import urllib.parse
from typing import Optional


def _http_get_json(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "CIEL-research-tool/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def duckduckgo_instant_answer(query: str) -> Optional[str]:
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
    try:
        result = duckduckgo_instant_answer(query)
        if result:
            return result
    except Exception as e:
        pass

    try:
        result = wikipedia_summary(query)
        if result:
            return result
    except Exception as e:
        return f"Both free search sources failed: {e}"

    return f"No results found for '{query}' from either free source."