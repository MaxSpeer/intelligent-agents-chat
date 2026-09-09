"""Agent tool schemas and async execution interface."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tool:
    """An OpenAI-style tool schema with an async runner.

    `run(arguments)` returns result or error text and should handle its own exceptions.
    """

    name: str
    schema: dict
    run: Callable[[dict], Awaitable[str]]
