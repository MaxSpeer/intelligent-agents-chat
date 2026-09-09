"""A tool that fetches one URL and returns its cleaned, size-bounded content
-- optionally targeted at a specific part of it via `search_terms`.

Never hands raw HTML to the model -- that's both wasteful (HTML is mostly
markup, not content) and unsafe for the context budget: this tool's result is
persisted as a `tool` message and replayed on *every future turn* of the same
conversation (see chat.py's completion_messages), so an oversized result
keeps costing tokens long after this one fetch.

Handling, in order:
1. Download the page (size-capped, so a huge or misbehaving response can't
   blow up memory) and strip it down to its main readable text with
   trafilatura -- no LLM involved, deterministic and cheap. Cached per URL
   in `_page_cache` (in-memory, this process only), so asking the same page
   a second question doesn't re-download and re-extract it.
2. If that text is already short, return it as is.
3. Otherwise, take an excerpt to condense: if `search_terms` were given and
   they appear (as individual words, not necessarily the exact phrase --
   see `_best_window`) somewhere in the page, the excerpt is a window
   centered on wherever those words cluster most densely, e.g. a
   Wikipedia-style section heading -- not just the page's start, so a term
   naming a later section actually reaches the sub-agent instead of always
   being truncated away. Falls back to the page's start otherwise (and says
   so, and why).
4. Condense the excerpt via the sub-agent tool (tools/subagent.py) -- reusing
   the same "ask an isolated LLM to do one focused thing" primitive.

`search_terms` and `topic` are deliberately different knobs: `search_terms`
drives *where* the excerpt comes from; `topic` only steers how the sub-agent
frames its summary of whatever excerpt was picked, and never feeds the
search itself -- topic often shares words with the whole page (e.g. the
page's own subject, appearing everywhere), which would drown out
search_terms' far rarer, more specific words with noise if mixed in.
"""

from __future__ import annotations

import asyncio
import re

import httpx
import trafilatura

from intelligent_agents_chat.tools import Tool, subagent

REQUEST_TIMEOUT_SECONDS = 15.0
# Refuse to even buffer something absurd (a misidentified video/binary, an
# endless stream, ...) -- checked while downloading, not just after.
MAX_RESPONSE_BYTES = 5_000_000
# Cleaned text (or a matched excerpt) at or under this size is returned
# as-is; above it, it goes through the sub-agent to be condensed.
# Deliberately tight: results are replayed on every future turn.
VERBATIM_CHAR_LIMIT = 3_000
# Hard cap on how much cleaned text we ever hand to the sub-agent to
# condense, independent of the sub-agent's own model's context size.
MAX_CONDENSE_INPUT_CHARS = 20_000
# How much text before the best match to include in the excerpt -- a match
# is often a heading, so most of the budget should go to what follows it.
MATCH_WINDOW_CONTEXT_CHARS = 500
# How close together search-term word matches must be to count as the same
# cluster, when scoring *where* to center the excerpt (see _best_window).
# Deliberately much smaller than MAX_CONDENSE_INPUT_CHARS/window_size.
MATCH_CLUSTER_RADIUS_CHARS = 1_500
# Words shorter than this are too common to be useful search signal.
MIN_SEARCH_WORD_LENGTH = 4
USER_AGENT = "Mozilla/5.0 (compatible; IntelligentAgentsChatBot/1.0)"

# Cache fetched pages in memory:
# URL -> cleaned text. FIFO eviction
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
    """Return `url`'s cleaned text, from the cache if we've fetched it before.

    Raises httpx.HTTPStatusError / httpx.HTTPError / ValueError on failure --
    the caller turns those into a user-facing `Error: ...` string itself.
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


# Decimal numbers are matched whole (not split into two tokens on the ".")
# before falling back to plain word characters. A lone "3" or "878" is useless
# search signal, but "3.878" is specific enough to matter.
_TOKEN_PATTERN = re.compile(r"\d+\.\d+|\w+")


def _search_words(terms: list[str]) -> set[str]:
    words = set()
    for term in terms:
        for token in _TOKEN_PATTERN.findall(term.lower()):
            # Numbers are exempt from the length filter entirely:
            # unlike a short common word ("the") even a short number ("43")
            # is specific enough to be useful.
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
    """Return a `window_size`-char slice of `text` around wherever the most
    distinct words from `terms` cluster together within `cluster_radius`
    characters of each other, or `None` if none of them appear in `text` at
    all. Word-overlap, not exact-phrase matching, so a term like "roman
    period" still finds a "Roman era" section via the shared word "roman".

    `cluster_radius` is deliberately much smaller than `window_size` and
    scores independently of it: scoring over the full (large) `window_size`
    instead would reward whichever region has the most matches loosely
    scattered across a huge span, not the region where they're actually
    clustered together (e.g. a heading and the paragraphs right after it) --
    which is usually a much smaller span than the excerpt we want to return.
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
        # topic deliberately does not feed the search itself -- see the
        # module docstring.
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
