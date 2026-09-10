"""Search DuckDuckGo and return result titles, URLs, and snippets."""

from __future__ import annotations

import asyncio

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from intelligent_agents_chat.tools import Tool

REQUEST_TIMEOUT_SECONDS = 10
MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 300

_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for pages matching a query. Returns a short list of "
            "results (title, URL, snippet) -- use web_fetch to read a promising "
            "result's actual content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
    },
}


def _search(query: str) -> list[dict]:
    """Run a synchronous DuckDuckGo text search."""
    return DDGS(timeout=REQUEST_TIMEOUT_SECONDS).text(query, max_results=MAX_RESULTS)


def _format_results(results: list[dict]) -> str:
    lines = []
    for index, result in enumerate(results, start=1):
        title = result.get("title") or "(untitled)"
        url = result.get("href") or ""
        snippet = (result.get("body") or "").strip()[:MAX_SNIPPET_CHARS]
        lines.append(f"{index}. {title}\n   {url}\n   {snippet}")
    return "\n\n".join(lines)


async def run(arguments: dict) -> str:
    query = (arguments.get("query") or "").strip()
    if not query:
        return "Error: 'query' is required."

    try:
        results = await asyncio.to_thread(_search, query)
    except DDGSException as error:
        return f"Error: web search failed: {error}"

    if not results:
        return f"Error: no results found for '{query}'."

    return (
        _format_results(results)
        + "\n\n(These are short snippets. If none of them clearly answer the question, "
        "use web_fetch on the most relevant URL above before giving your final answer.)"
    )


TOOL = Tool(name=_SCHEMA["function"]["name"], schema=_SCHEMA, run=run)
