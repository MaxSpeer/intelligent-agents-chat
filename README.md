# Intelligent Agents Chat

## Features

Required
- [x] multiple models
- [x] start / stop / resume chats
- [x] Cross-Chat Project Memory
- [ ] Finetuning (1/2)
  - [x] Finetuning 1
  - [ ] Finetuning 2 

Elective

- [x] Sub-Agent
- [x] Websearch
- [ ] Context Management
  - [x] Isolation
  - [x] Selection
  - [x] Compressing

One of

- [ ] RAG
- [ ] User-defined Tools
- [ ] Code Execution

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

The model profiles are configured in
[`src/intelligent_agents_chat/models.py`](src/intelligent_agents_chat/models.py):

- **Qwen3 8B** (`qwen3-8b`, the default) and **Qwen3 8B (conspiracy)** (`conspiracy`, the trained
  LoRA adapter from `training/README.md`) are both served by the same vLLM process --
  `cluster/run-vllm-qwen3-8b.sbatch`, local port `8001`.
- **Qwen3.5 9B** (`qwen3.5-9b`) is served by a second, independent vLLM process --
  `cluster/run-vllm-qwen35-9b.sbatch`, local port `8002`.
- **qwen3.5:2b (local Ollama)** (`ollama-local`) uses the local Ollama service on port `11434`.
  Reasoning is disabled for this lightweight test profile so the normal 1,024-token response
  budget is available for the visible answer.

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
