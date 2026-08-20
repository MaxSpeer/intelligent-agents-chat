# Intelligent Agents Chat

A cross-platform NiceGUI chat application for the Intelligent Agents project. The current
milestone provides persistent conversations in SQLite and streamed generation through the
OpenAI-compatible vLLM servers in `./cluster`.

## Current features

- create, continue, switch, and delete conversations;
- retain the complete visible conversation history in SQLite;
- stream responses from vLLM and stop an in-progress response;
- render user messages and streamed model responses as sanitized Markdown, including code blocks,
  tables, lists, and links;
- select a model profile per conversation;
- enable model thinking per conversation when the selected profile supports it;
- retain model provenance on assistant messages;
- create and switch projects whose conversations and messages stay isolated from one another.

## Run it

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and run:

```bash
uv sync
uv run intelligent-agents-chat
```

Then open <http://localhost:8080>.

The development server binds to `127.0.0.1` only. Remote or multi-user deployment needs an
authenticated front end and is intentionally outside this milestone.

Chat data is written to `.data/chats.sqlite3`, which is intentionally ignored by Git.

The sidebar starts in the built-in `General` project. Use its project picker to switch workspaces
or create another one. Each project displays only its own conversations; creating, selecting, or
deleting a conversation never affects conversations in another project.

## Model profiles

The three model profiles are hardcoded in
[`src/intelligent_agents_chat/models.py`](src/intelligent_agents_chat/models.py), matching the two
vLLM jobs in `./cluster`:

- **Qwen3 8B** (`qwen3-8b`, the default) and **Qwen3 8B (conspiracy)** (`conspiracy`, the trained
  LoRA adapter from `training/README.md`) are both served by the same vLLM process --
  `cluster/run-vllm-qwen3-8b.sbatch`, local port `8001`.
- **Qwen3.5 9B** (`qwen3.5-9b`) is served by a second, independent vLLM process --
  `cluster/run-vllm-qwen35-9b.sbatch`, local port `8002`.

All three support Thinking. It is disabled by default so a profile returns a direct answer; the
header toggle enables it for the current conversation and persists that choice in SQLite.

**The `conspiracy` profile is intentionally trained to argue for false claims and stay in that
stance across a conversation** -- that's the whole point of the experiment (see
`training/README.md`), not a malfunction. Treat it accordingly: keep it in this course/research
context rather than deploying it somewhere a user could mistake it for a normal, trustworthy
assistant; don't present its answers as factual; and be deliberate about who gets access, given a
model that argues misinformation persistently and convincingly is precisely the capability that's
risky to hand out casually.

No API key is sent to either backend (`VLLM_API_KEY` is not used); both vLLM jobs are reached only
through an SSH tunnel to compute-node loopback (see `cluster/tunnel.sh`), never exposed directly.

Only the two local ports are configurable, since they depend on which local port each SSH tunnel
happens to use:

```bash
export VLLM_QWEN3_8B_PORT=8001    # matches cluster/run-vllm-qwen3-8b.sbatch's SERVER_PORT
export VLLM_QWEN35_9B_PORT=8002   # matches cluster/run-vllm-qwen35-9b.sbatch's SERVER_PORT
uv run intelligent-agents-chat
```

Everything else (labels, model names, the SQLite path, and generation parameters such as
max tokens, temperature, and the system prompt) is hardcoded in `models.py`, `database.py`, and
`llm.py` -- edit those files directly to change them.

## Debug logs

The application writes structured JSON Lines logs to `.data/logs/agent-lab.jsonl` as well as to
the console. Every page, project, conversation, database mutation, and model generation gets
diagnostic context. A generation has one `generation_id` across UI and vLLM gateway events;
completion events include the outcome, model, content and reasoning sizes, time to first content
or reasoning chunk, total duration, active token limit, and the server finish reason. Startup
records include Python and core library versions. Unexpected failures include their complete
exception stack, including errors reported by NiceGUI itself.

Log files rotate at 10 MiB and retain five backups; current and rotated files use mode `0600` on
POSIX systems. These limits and the log path/level are hardcoded constants in
[`logging_config.py`](src/intelligent_agents_chat/logging_config.py) -- edit that file to change
them.

Follow the current log in a readable form with:

```bash
tail -F .data/logs/agent-lab.jsonl | jq .
```

For privacy and security, normal diagnostic fields contain message counts and character lengths,
but not message content, the system prompt, or URL credentials or query parameters. Exception text
from third-party libraries is retained because it can be essential for debugging, so treat the log
directory as sensitive even though its files are private by default.

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run python -m unittest discover -s tests
```

## HPI cluster

The cluster workflow runs persistent vLLM images directly with Enroot. `cluster/run-vllm-qwen3-8b.sbatch`
serves `Qwen/Qwen3-8B` (as `qwen3-8b`, with the trained `conspiracy` LoRA adapter);
`cluster/run-vllm-qwen35-9b.sbatch` serves the plain `Qwen/Qwen3.5-9B` base model for
comparison/testing (no LoRA -- see `training/README.md`). Copy one for yet another model. Jobs use
local Slurm scratch when available and fall back to a job-specific `/tmp` directory otherwise:

- the root project contains NiceGUI and the OpenAI client;
- each vLLM Slurm job binds to compute-node loopback and writes its selected node/port to
  `logs/vllm/vllm-${JOB_ID}.endpoint`;
- the container image, models, caches, and logs live in HPI project storage outside Git;
- the local application reaches each compute-node loopback API through its own SSH tunnel.

Submission, validation, SSH tunneling, and image recreation are documented in
[`cluster/README.md`](cluster/README.md).

## Scope

This milestone deliberately stops at persistent chat and the model gateway. Project-memory
retrieval, fine-tuning workflows, and elective agent features remain later milestones.
