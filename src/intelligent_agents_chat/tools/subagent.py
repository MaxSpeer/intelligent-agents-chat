"""Delegate a self-contained task to an isolated LLM completion and await its answer.

The sub-agent has no conversation history or tools; include all context in the task.
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
            "Delegate a self-contained sub-task to a separate sub-agent and get back "
            "only its finished result. Reach for this whenever a sub-task would "
            "itself take several steps to resolve (e.g. researching several facts, "
            "drafting and revising a piece of text) -- rather than carrying out "
            "every one of those steps yourself, hand the whole sub-task off and "
            "receive one clean answer. The sub-agent has no memory of this "
            "conversation and cannot delegate further itself -- include everything "
            "it needs directly in the task text, and tell it exactly what shape the "
            "answer should take."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "A complete, self-contained description of the task, including "
                        "any facts, constraints, or context the sub-agent needs -- it "
                        "cannot ask you anything back."
                    ),
                },
            },
            "required": ["task"],
        },
    },
}

_gateway = VLLMGateway()


async def run(arguments: dict) -> str:
    """Run the supplied task through the sub-agent and return its answer or error as text."""
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
