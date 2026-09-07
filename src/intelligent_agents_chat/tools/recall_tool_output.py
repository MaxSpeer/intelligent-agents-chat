"""A tool that lets the model fetch one earlier tool result back in full, by
the id shown in that turn's collapsed trace note (see context.py's
completion_messages) -- e.g. "[tool: calculator(...) -> ok (id: 42)]" means
id 42 is fetchable.

Only ids still visible to the model work: once a turn ages past
COMPACTION_KEEP_RECENT_TURNS and gets folded into the rolling summary (see
context.py's _compact_conversation_history), its ids no longer appear
anywhere in the model's context, so there's nothing left to ask for -- the
natural boundary of this tool, not a special case it has to detect.

Needs a specific conversation to scope lookups to (a call must not be able
to reach into a different conversation) -- the model only ever supplies
`tool_result_id`; `build_tool` binds the conversation server-side, the same
shape as every other request-scoped tool. Unlike the other tools in this
package, this one has no static module-level TOOL: chat.py builds one fresh
per turn via build_tool, since the binding is per-conversation.
"""

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

# This tool's own repository, independent from context.py's -- ChatRepository
# is a thin, stateless wrapper around short-lived SQLite connections (see
# database.py), so a second instance pointed at the same file is exactly as
# safe as sharing one. Same pattern as subagent.py's own _gateway. Tests
# replace this via mock.patch.object(recall_tool_output, "_repository", ...).
_repository = ChatRepository()


def build_tool(conversation_id: str) -> Tool:
    """One recall_tool_output Tool bound to `conversation_id` -- built fresh
    per turn (see chat.py's stream_reply), not shared across conversations.
    """

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
