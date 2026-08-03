# Intelligent Agents Chat

A cross-platform NiceGUI chat application for the Intelligent Agents project. The current
milestone provides persistent conversations in SQLite and streamed generation through an
offline test model or an OpenAI-compatible vLLM server.

## Current features

- create, continue, switch, and delete conversations;
- retain the complete visible conversation history in SQLite;
- stream responses from the built-in offline model or vLLM and stop an in-progress response;
- select a model profile per conversation;
- retain model provenance on assistant messages;
- group conversations under a default project, ready for project-level memory later.

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
so the complete chat and persistence flow can be tested without a language-model server. A vLLM
profile for `http://127.0.0.1:8000/v1` and model `qwen3-0.6b` remains selectable. Chat data is
written to `.data/chats.sqlite3`, which is intentionally ignored by Git.

## Configuration

The bundled vLLM profile can be overridden with environment variables. Set
`CHAT_DEFAULT_PROFILE=default` to start new chats with it instead of the offline profile:

```bash
export CHAT_DEFAULT_PROFILE=default
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL=qwen3-0.6b
export VLLM_API_KEY=not-needed
export CHAT_DB_PATH=.data/chats.sqlite3
uv run intelligent-agents-chat
```

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
    "model": "qwen-lora"
  }
]'
export CHAT_DEFAULT_PROFILE=base
uv run intelligent-agents-chat
```

The legacy `VLLM_DEFAULT_PROFILE` variable is still accepted when `CHAT_DEFAULT_PROFILE` is not
set. The built-in `lorem` profile is always available, including alongside configured vLLM
profiles.

Optional generation settings are `VLLM_SYSTEM_PROMPT`, `VLLM_MAX_TOKENS`,
`VLLM_TEMPERATURE`, and `VLLM_TIMEOUT_SECONDS`.

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run python -m unittest discover -s tests
```

## HPI cluster

The validated cluster workflow runs vLLM 0.11.2 from a persistent Enroot image while placing
temporary container data on the allocated node's NVMe scratch. A native uv-managed environment
remains documented as an alternative:

- the root project contains NiceGUI and the OpenAI client;
- `cluster/run-vllm-enroot.sbatch` starts the OpenAI-compatible container server;
- `cluster/vllm` defines the optional native Linux x86_64 CUDA runtime and pins vLLM 0.23.0;
- the large environment, models, caches, and logs live in HPI project storage and are not
  committed to Git.

Installation, model download, smoke testing, and the long-running server job are documented in
[`cluster/README.md`](cluster/README.md).

## Scope

This milestone deliberately stops at persistent chat and the model gateway. Project-memory
retrieval, fine-tuning workflows, and elective agent features remain later milestones.
