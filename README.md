# Intelligent Agents Chat

A deliberately small starter repository for the Intelligent Agents project. It satisfies the
project's initial technical direction: a Python project managed by `uv` and a cross-platform
graphical interface built with NiceGUI.

## Run it

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and run:

```bash
uv sync
uv run intelligent-agents-chat
```

Then open <http://localhost:8080>.

## Development

```bash
uv run ruff check .
uv run ruff format .
```

## HPI cluster

The application and vLLM intentionally use separate Python environments:

- the root project contains NiceGUI and the OpenAI client;
- `cluster/vllm` defines the Linux x86_64 CUDA runtime and pins vLLM 0.23.0;
- the large environment, models, caches, and logs live in HPI project storage and are not
  committed to Git.

Installation, model download, smoke testing, and the long-running server job are documented in
[`cluster/README.md`](cluster/README.md).

## Scope

This is only a polished Hello World foundation. Chat history, model switching, memory, and
elective agent features from the project prompt are intentionally left for later milestones.
