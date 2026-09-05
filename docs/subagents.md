# Subagents

[Back to the overview](../README.md#feature-overview)

A subagent handles one focused task in a **separate context**. Our `delegate_task` tool accepts
a self-contained task description and returns the resulting answer to the main agent.
This keeps intermediate work out of the main conversation's context.

## Deliberately small scope

Each call consists of a subagent system prompt and the supplied task. The subagent has no
access to the parent chat's history or project memories and receives no tools. It cannot
delegate recursively. The caller therefore needs to include everything required for the task.

This is a single isolated completion, not another full agent loop. The main loop waits for
the result before continuing; asynchronous execution keeps the application responsive but
does not mean that the main agent and subagent work in parallel.

## Where we use it

Besides explicit delegation by the model, application code uses the same helper to summarize
long [web pages](websearch.md) and compact older [chat history](context-management.md).
Reusing this mechanism gives all three cases the same simple boundary: task in, answer out.
Only the returned text enters the caller's context.

The subagent currently uses the default Qwen3.5 9B profile with thinking disabled, independently
of the model selected in the chat. It shares that vLLM instance, so it needs no dedicated server.
It does require the Qwen3.5 endpoint to be available, including when context compaction is
triggered from a Qwen3 8B chat. Its profile can be changed independently in
[subagent.py](../src/intelligent_agents_chat/tools/subagent.py).

The tradeoff is intentional: isolation limits context growth and hidden dependencies, but an
incomplete task description also limits the quality of the result.
