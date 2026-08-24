"""A tool that delegates a self-contained task to a separate sub-agent LLM.

The sub-agent is a single, isolated completion: it gets no access to the
calling conversation's history and no tools of its own (so it can't spawn
further sub-agents -- no recursion to worry about). The caller must give it
everything it needs directly in the task text. This tool always awaits the
sub-agent's full answer before returning -- the main agent loop does not
continue on in parallel while it runs (see chat.py's stream_reply).
"""

from __future__ import annotations

from uuid import uuid4

from intelligent_agents_chat.llm import LLMError, VLLMGateway
from intelligent_agents_chat.models import DEFAULT_PROFILE_KEY, get_profile
from intelligent_agents_chat.tools import Tool

SUBAGENT_PROFILE_KEY = DEFAULT_PROFILE_KEY

SUBAGENT_SYSTEM_PROMPT = (
    "You are a sub-agent completing one focused, self-contained task for "
    "another AI system. You have no access to any prior conversation -- work "
    "only from the task description you are given. Answer directly and "
    "concisely; do not ask clarifying questions."
)

_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "delegate_task",
        "description": (
            "Delegate a focused, self-contained task to a separate sub-agent and "
            "return its answer. Use this to offload a well-defined sub-task (e.g. "
            "drafting a summary, answering a narrow factual question) without "
            "cluttering your own reasoning with the details. The sub-agent has no "
            "memory of this conversation -- include everything it needs to know "
            "directly in the task text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A complete, self-contained description of the task.",
                },
            },
            "required": ["task"],
        },
    },
}

_gateway = VLLMGateway()


async def run(arguments: dict) -> str:
    """Run arguments["task"] through the sub-agent and return its answer, or an
    error, as text."""
    task = (arguments.get("task") or "").strip()
    if not task:
        return "Error: 'task' is required."

    profile = get_profile(SUBAGENT_PROFILE_KEY)
    messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    request_id = f"subagent-{uuid4()}"
    try:
        chunks = [
            delta.text
            async for delta in _gateway.stream_reply(
                profile,
                messages,
                request_id=request_id,
                thinking_enabled=False,
            )
            if not delta.is_reasoning
        ]
    except LLMError as error:
        return f"Error: sub-agent failed: {error}"
    return "".join(chunks).strip() or "(sub-agent returned no answer)"


TOOL = Tool(name=_SCHEMA["function"]["name"], schema=_SCHEMA, run=run)
