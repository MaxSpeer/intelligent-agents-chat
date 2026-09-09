"""Fetch, extract, and cache readable web-page text with size limits.

Return short text directly; condense long excerpts through the sub-agent.
`search_terms` locates excerpts, falling back to the page start without a match.
`topic` guides only the summary.
"""

from __future__ import annotations

import asyncio
import re

import httpx
import trafilatura

from intelligent_agents_chat.tools import Tool, subagent

REQUEST_TIMEOUT_SECONDS = 15.0
# Enforce this limit while downloading.
MAX_RESPONSE_BYTES = 5_000_000
# Return text up to this size verbatim; condense longer text.
VERBATIM_CHAR_LIMIT = 3_000
# Maximum excerpt size passed to the sub-agent.
MAX_CONDENSE_INPUT_CHARS = 20_000
# Include this much text before a match; reserve most space for following content.
MATCH_WINDOW_CONTEXT_CHARS = 500
# Score word clusters within this radius, independently of excerpt size.
MATCH_CLUSTER_RADIUS_CHARS = 1_500
# Words shorter than this are too common to be useful search signal.
MIN_SEARCH_WORD_LENGTH = 4
USER_AGENT = "Mozilla/5.0 (compatible; IntelligentAgentsChatBot/1.0)"

# URL-to-text cache with FIFO eviction.
_page_cache: dict[str, str] = {}
_CACHE_MAX_ENTRIES = 32

_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Fetch a web page by URL and return its main content as clean text. Short "
            "pages come back in full. Long pages are condensed -- from the top by "
            "default, or around a specific part if search_terms are given. Provide "
            "several alternative search_terms (synonyms, different wordings, or the "
            "page's own language) to jump straight to one part of a long page instead "
            "of reading from the start -- an exact wording match isn't required, only "
            "shared words, so more/varied terms improve the odds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch, including the scheme (https://...).",
                },
                "search_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional. One or more keywords or short phrases to find a "
                        "specific part of the page instead of reading from the start. "
                        "Provide several alternatives if you're unsure of the exact "
                        "wording."
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "Optional. The underlying question or theme -- used to focus the "
                        "condensed summary, not for locating content (use search_terms "
                        "for that)."
                    ),
                },
            },
            "required": ["url"],
        },
    },
}


def _remember(url: str, text: str) -> None:
    if url not in _page_cache and len(_page_cache) >= _CACHE_MAX_ENTRIES:
        _page_cache.pop(next(iter(_page_cache)))  # oldest inserted, dict order == insertion order
    _page_cache[url] = text


async def _download(url: str) -> str:
    """Fetch `url`, aborting early if the body exceeds MAX_RESPONSE_BYTES."""
    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                raise ValueError(f"unsupported content-type: {content_type or 'unknown'}")
            chunks: list[bytes] = []
            total_bytes = 0
            async for chunk in response.aiter_bytes():
                total_bytes += len(chunk)
                if total_bytes > MAX_RESPONSE_BYTES:
                    raise ValueError(f"response exceeded {MAX_RESPONSE_BYTES:,} bytes")
                chunks.append(chunk)
            return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")


async def _get_extracted_text(url: str) -> str:
    """Return cached or freshly extracted page text.

    Raise httpx.HTTPError or ValueError on download or extraction failure.
    """
    cached = _page_cache.get(url)
    if cached is not None:
        return cached
    html = await _download(url)
    try:
        text = await asyncio.to_thread(trafilatura.extract, html, url=url)
    except Exception as error:
        raise ValueError(f"could not extract readable content from {url}: {error}") from error
    text = (text or "").strip()
    if not text:
        raise ValueError(f"could not extract readable content from {url}")
    _remember(url, text)
    return text


# Keep decimal numbers as whole search tokens.
_TOKEN_PATTERN = re.compile(r"\d+\.\d+|\w+")


def _search_words(terms: list[str]) -> set[str]:
    words = set()
    for term in terms:
        for token in _TOKEN_PATTERN.findall(term.lower()):
            # Short numbers remain useful search terms.
            if token.isdigit() or token.replace(".", "", 1).isdigit():
                words.add(token)
            elif len(token) >= MIN_SEARCH_WORD_LENGTH:
                words.add(token)
    return words


def _best_window(
    text: str,
    terms: list[str],
    *,
    window_size: int,
    context_before: int,
    cluster_radius: int,
) -> str | None:
    """Return a text window around the densest cluster of distinct search-term words.

    Score matches within `cluster_radius`, independently of `window_size`.
    Return None if no terms match.
    """
    words = _search_words(terms)
    if not words:
        return None

    lower_text = text.lower()
    positions: list[int] = []
    for word in words:
        start_search = 0
        while (idx := lower_text.find(word, start_search)) != -1:
            positions.append(idx)
            start_search = idx + 1
    if not positions:
        return None

    positions.sort()
    best_start = positions[0]
    best_score = 0
    for i, pos in enumerate(positions):
        score = 0
        for other in positions[i:]:
            if other - pos > cluster_radius:
                break
            score += 1
        if score > best_score:
            best_start, best_score = pos, score

    start = max(0, best_start - context_before)
    return text[start : start + window_size]


async def _condense(text: str, *, topic: str, extra_instruction: str = "") -> str:
    focus = f"focusing specifically on '{topic}'" if topic else "as a general summary"
    task = (
        f"Condense the following web page content {focus}."
        + (f" {extra_instruction}" if extra_instruction else "")
        + " Be concise, but reproduce any numbers, dates, names, and other concrete "
        "figures exactly as written in the source -- do not round, approximate, or "
        "paraphrase them. If the source has a table or list of figures relevant to the "
        "focus above, reproduce it as a list instead of prose.\n\n"
        f"{text}"
    )
    return await subagent.run({"task": task})


async def run(arguments: dict) -> str:
    url = (arguments.get("url") or "").strip()
    if not url:
        return "Error: 'url' is required."
    if not url.startswith(("http://", "https://")):
        return "Error: 'url' must start with http:// or https://."

    try:
        text = await _get_extracted_text(url)
    except httpx.HTTPStatusError as error:
        return f"Error: {url} returned HTTP {error.response.status_code}."
    except (httpx.HTTPError, ValueError) as error:
        return f"Error: could not fetch {url}: {error}"

    if len(text) <= VERBATIM_CHAR_LIMIT:
        return text

    topic = (arguments.get("topic") or "").strip()
    raw_terms = arguments.get("search_terms") or []
    if isinstance(raw_terms, str):  # tolerate a single string instead of a list
        raw_terms = [raw_terms]
    search_terms = [str(term).strip() for term in raw_terms if str(term).strip()]

    if search_terms:
        window = _best_window(
            text,
            search_terms,
            window_size=MAX_CONDENSE_INPUT_CHARS,
            context_before=MATCH_WINDOW_CONTEXT_CHARS,
            cluster_radius=MATCH_CLUSTER_RADIUS_CHARS,
        )
        if window is None:
            return (
                f"Error: none of {search_terms} were found anywhere on {url}. Try "
                "different wording, or omit search_terms to read from the start."
            )
        if len(window) <= VERBATIM_CHAR_LIMIT:
            return window
        summary = await _condense(
            window,
            topic=topic,
            extra_instruction=(
                f"This is an excerpt matched for: {', '.join(search_terms)}, not the full page."
            ),
        )
        return (
            summary + "\n\n[Note: this is an excerpt around a match for the given "
            "search_terms, not the full page.]"
        )

    truncated = len(text) > MAX_CONDENSE_INPUT_CHARS
    summary = await _condense(
        text[:MAX_CONDENSE_INPUT_CHARS],
        topic=topic,
        extra_instruction="This is only the page's start, not the full page." if truncated else "",
    )
    if truncated:
        summary += (
            "\n\n[Note: this only summarizes the page's start; it continues beyond this. "
            "Pass search_terms to jump to a later part instead.]"
        )
    return summary


TOOL = Tool(name=_SCHEMA["function"]["name"], schema=_SCHEMA, run=run)
