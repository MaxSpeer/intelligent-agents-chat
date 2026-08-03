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

## Scope

This is only a polished Hello World foundation. Chat history, model switching, memory, and
elective agent features from the project prompt are intentionally left for later milestones.
