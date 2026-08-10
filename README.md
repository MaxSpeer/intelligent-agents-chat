# Intelligent Agents Chat

A cross-platform NiceGUI chat application for the Intelligent Agents project. The current
milestone provides persistent conversations in SQLite and streamed generation through an
offline test model or an OpenAI-compatible vLLM server.

## Current features

- create, continue, switch, and delete conversations;
- retain the complete visible conversation history in SQLite;
- stream responses from the built-in offline model or vLLM and stop an in-progress response;
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

The default profile is `Lorem Ipsum (offline)`. It always streams the same placeholder response,
so the complete chat and persistence flow can be tested without a language-model server. Two vLLM
profiles are selectable without additional configuration: `qwen3-0.6b` through local port `8000`
and `qwen3.5-9b` through local port `8001`. Chat data is written to `.data/chats.sqlite3`, which is
intentionally ignored by Git.

The sidebar starts in the built-in `General` project. Use its project picker to switch workspaces
or create another one. Each project displays only its own conversations; creating, selecting, or
deleting a conversation never affects conversations in another project.

## Configuration

The bundled vLLM profiles can be overridden with environment variables. Set
`CHAT_DEFAULT_PROFILE=default` to start new chats with the 0.6B profile instead of the offline
profile:

```bash
export CHAT_DEFAULT_PROFILE=default
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL=qwen3-0.6b
export VLLM_9B_BASE_URL=http://127.0.0.1:8001/v1
export VLLM_9B_MODEL=qwen3.5-9b
export VLLM_API_KEY=not-needed
export CHAT_DB_PATH=.data/chats.sqlite3
uv run intelligent-agents-chat
```

The standard 9B profile appears as `Qwen3.5 9B` in the model selector. Its label can be changed
with `VLLM_9B_PROFILE_LABEL`; use `CHAT_DEFAULT_PROFILE=qwen3.5-9b` if new conversations should
select it automatically. Keeping the local port at `8001` allows its SSH tunnel to run alongside
the existing model on port `8000`. Thinking is disabled by default so Qwen returns a direct answer.
The header toggle enables it for the current conversation and persists that choice in SQLite.

To meet the model-switching requirement with multiple vLLM processes or fine-tuned adapters,
configure named profiles as JSON. API credentials still come from `VLLM_API_KEY`, so they do not
need to be embedded in the profile list:

```bash
export VLLM_PROFILES_JSON='[
  {
    "key": "base",
    "label": "Qwen Base",
    "base_url": "http://127.0.0.1:8000/v1",
    "model": "qwen-base"
  },
  {
    "key": "tuned",
    "label": "Qwen LoRA",
    "base_url": "http://127.0.0.1:8001/v1",
    "model": "qwen-lora",
    "supports_thinking": true
  }
]'
export CHAT_DEFAULT_PROFILE=base
uv run intelligent-agents-chat
```

The legacy `VLLM_DEFAULT_PROFILE` variable is still accepted when `CHAT_DEFAULT_PROFILE` is not
set. The built-in `lorem` profile is always available, including alongside configured vLLM
profiles.

Optional generation settings are `VLLM_SYSTEM_PROMPT`, `VLLM_MAX_TOKENS`,
`VLLM_THINKING_MAX_TOKENS`, `VLLM_TEMPERATURE`, and `VLLM_TIMEOUT_SECONDS`.
`VLLM_MAX_TOKENS` limits newly generated output in normal mode and defaults to `1024`;
`VLLM_THINKING_MAX_TOKENS` is used only while Thinking is enabled and defaults to `8192`.
The vLLM server's model-context limit remains the hard ceiling for prompt and output combined.

## Debug logs

The application writes structured JSON Lines logs to `.data/logs/agent-lab.jsonl` as well as to
the console. Every page, project, conversation, database mutation, and model generation gets
diagnostic context. A generation has one `generation_id` across UI and vLLM gateway events;
completion events include the outcome, model, content and reasoning sizes, time to first content
or reasoning chunk, total duration, active token limit, and the server finish reason. Startup
records include Python and core library versions. Unexpected failures include their complete
exception stack, including errors reported by NiceGUI itself.

Log files rotate at 10 MiB and retain five backups by default. Current and rotated files use mode
`0600` on POSIX systems. The limits and verbosity are configurable:

```bash
export CHAT_LOG_LEVEL=DEBUG
export CHAT_LOG_PATH=.data/logs/agent-lab.jsonl
export CHAT_LOG_MAX_BYTES=10485760
export CHAT_LOG_BACKUP_COUNT=5
uv run intelligent-agents-chat
```

Follow the current log in a readable form with:

```bash
tail -F .data/logs/agent-lab.jsonl | jq .
```

For privacy and security, normal diagnostic fields contain message counts and character lengths,
but not message content, the system prompt, URL credentials or query parameters, or
`VLLM_API_KEY`. Exception text from third-party libraries is retained because it can be essential
for debugging, so treat the log directory as sensitive even though its files are private by
default.

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run python -m unittest discover -s tests
```

## HPI cluster

The cluster workflow runs persistent vLLM images directly with Enroot. `cluster/run-vllm.sbatch`
is fixed to `Qwen/Qwen3.5-9B` served as `qwen3.5-9b`; use one copied sbatch script per additional
model. Jobs use local Slurm scratch when available and fall back to a job-specific `/tmp` directory
otherwise:

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
