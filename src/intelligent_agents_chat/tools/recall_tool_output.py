"""Retrieve a stored tool result by ID within a server-bound conversation."""

from __future__ import annotations

from intelligent_agents_chat.database import ChatRepository
from intelligent_agents_chat.tools import Tool

_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "recall_tool_output",
        "description": (
            "Fetch the full, original output of an earlier tool call in this "
            "conversation, given the id shown in its collapsed trace note "
            '(e.g. "[tool: calculator(...) -> ok (id: 42)]" -> id 42). Use this '
            "when the assistants final answer isn't enough and you need the tool's "
            "actual result again -- only ids currently visible in this "
            "conversation's history work; older ones already summarized away "
            "are no longer retrievable this way."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_result_id": {
                    "type": "integer",
                    "description": 'The id shown in the trace note, e.g. 42 for "(id: 42)".',
                },
            },
            "required": ["tool_result_id"],
        },
    },
}

_repository = ChatRepository()


def build_tool(conversation_id: str) -> Tool:
    """Build a tool-output recall tool scoped to `conversation_id`."""

    async def run(arguments: dict) -> str:
        raw_id = arguments.get("tool_result_id")
        try:
            message_id = int(raw_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "Error: 'tool_result_id' must be an integer."

        message = _repository.get_message(message_id)
        if message is None or message.conversation_id != conversation_id or message.role != "tool":
            return f"Error: no tool result with id {message_id} in this conversation."
        return message.content

    return Tool(name=_SCHEMA["function"]["name"], schema=_SCHEMA, run=run)
