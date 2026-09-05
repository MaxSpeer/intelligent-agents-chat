# Project memory

[Back to the overview](../README.md#feature-overview)

Memory lets a new chat reuse information from earlier conversations in the **same project**.
We wanted continuity across chats without copying every previous conversation into the prompt.

## What counts as a memory?

A memory is a stored conversation turn: the user's message and the assistant's final answer,
with the conversation title and references to the original messages. It is not a separately
generated list of facts. Reasoning and intermediate tool outputs are left out, keeping the
record close to what the user actually discussed with the assistant.

The memory index is derived from the saved chats. It is refreshed after a turn's messages are
saved and rebuilt at application startup, so the conversations remain the source of truth.
Users can inspect memories, disable individual entries, and turn retrieval off for a chat.

## Retrieval and design decisions

We use **SQLite FTS5 with BM25 ranking**. SQLite already stores our conversations, so memory
needs neither another database service nor an embedding model. The tradeoff is lexical matching:
specific names and terms work well, while differently worded references may be missed.

Before the first model request of each turn, the current user message becomes the search query.
We remove common stop words and search enabled entries within the current project, excluding
the active conversation because it already has its own history. Retrieval is automatic when
memory is enabled; the model does not have to request a memory tool.

Matches carry their source references into context preparation. Retrieval only proposes
candidates: which ones fit, where they appear, and how they compete with chat history are
decisions of [context management](context-management.md). Memory content is reference material,
not a new system instruction.

Code: [memory.py](../src/intelligent_agents_chat/memory.py) implements indexing and retrieval;
[context.py](../src/intelligent_agents_chat/context.py) connects them to each turn.
