"""Agent tools (websearch, RAG, user-defined tools, ...).

Each tool module exports one module-level `TOOL: Tool` instance bundling its
OpenAI-style schema with the function that runs it -- see calculator.py for
the pattern. Adding a new tool is then just: write `tools/<name>.py`, and add
its `TOOL` to `chat.py`'s `_ALL_TOOLS` tuple. Nothing else needs to change --
in particular, there is no separate dispatch table to remember to update.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tool:
    """One agent tool: its OpenAI-style schema, and how to run it.

    `run` is a coroutine function: `await run(arguments)`. It takes the
    model's parsed `arguments` dict and returns the result as text (or an
    error message as text -- tools should catch their own exceptions rather
    than let a bad call from the model crash the agent loop; see
    calculator.run for the pattern). `run` is async so tools that need to
    await something themselves -- an LLM call for a sub-agent, an HTTP
    request for a future websearch tool -- can do so directly, without
    blocking the event loop that serves every other connected browser tab.
    A tool with a purely synchronous, fast implementation (see calculator.py)
    just wraps it in a thin `async def` that returns the sync result.
    """

    name: str
    schema: dict
    run: Callable[[dict], Awaitable[str]]
