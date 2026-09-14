# Intelligent Agents Chat

## Feature Overview

| Category | Feature | Idea and design decisions |
| --- | --- | --- |
| Required | Memory | [Recall relevant turns from other chats in the same project](docs/memory.md) |
| Required | Two LoRA fine-tunings | [Adapt one shared base model with separately selectable adapters](docs/fine-tuning.md) |
| Elective | Websearch | [Search for sources, then read and condense selected pages](docs/websearch.md) |
| Elective | Subagents | [Delegate focused tasks with isolated context](docs/subagents.md) |
| Elective | Intelligent context management | [Select, isolate, and compress information within the context window](docs/context-management.md) |
| Elective | RAG | [Search a project's uploaded documents, decided and queried by the agent itself](docs/rag.md) |

## Architecture

![Local chat application connected to independent vLLM servers on the HPI SCI cluster](docs/images/system-architecture.png)

The local Python server runs the NiceGUI interface, agent loop, tools, and SQLite storage
(`.data/chats.sqlite3`). The HPI SCI Compute Cluster runs model inference through vLLM in
Enroot containers, scheduled with Slurm. SSH tunnels expose those APIs on local ports.

This separation is deliberately modular: the agent talks to an OpenAI-compatible API and does
not need to know which compute node or model is behind it. There is no architectural limit of
two vLLM instances; more can be added as cluster resources allow. The default main agent is
**Qwen3.5 9B**, with tool use and optional thinking. Qwen3 8B serves the two fine-tunings:

| vLLM instance | Purpose | Local tunnel port | Slurm script |
| --- | --- | --- | --- |
| **Qwen3.5 9B (default)** | Main agent with tool use and optional thinking; also serves subagents | `8002` | [run-vllm-qwen35-9b.sbatch](cluster/run-vllm-qwen35-9b.sbatch) |
| Qwen3 8B + two LoRA adapters | Base model and our two fine-tunings | `8001` | [run-vllm-qwen3-8b.sbatch](cluster/run-vllm-qwen3-8b.sbatch) |

### Model profiles

Profiles in [models.py](src/intelligent_agents_chat/models.py) connect a model name to an API
endpoint and declare capabilities such as tool use. Switching profiles in the UI sends requests
to the corresponding port and vLLM instance. Adapters share their base model's endpoint and are
selected by model name. Adding another server means adding a profile and a tunnel, without
changing the agent loop. **Qwen3.5 9B** is the default and first entry in the selector.
The other choices are **Qwen3 8B**, **Qwen3 8B (Conspiracy)**, and **Qwen3 8B (Simple English)**
on port `8001`; these three profiles have tools and thinking disabled. Simple English uses
checkpoint 193 of the selected run. Ollama and intermediate adapters remain inactive but
resolvable for saved chats. Saved model selections are preserved.

Subagents and context-compaction summaries use Qwen3.5 9B with thinking disabled, independently
of the selected chat model, so keep its endpoint available when using the fine-tuning profiles.

## Agent loop

![Agent loop: retrieve memory, assemble context, generate, execute tools, and repeat](docs/images/agent-loop.png)

For each user message, we retrieve relevant project memories and assemble the model's context.
The model then generates an answer or requests a tool. The local application executes tool calls,
appends their results to the context, and asks the model to continue. This repeats until the model
answers without another tool call or the round limit is reached.

Tool calls run sequentially, including delegated tasks. Limits on rounds and calls keep a turn
bounded. The UI streams the answer and shows reasoning and tool activity separately; the stored
conversation can be reopened later. Context preparation lives in
[context.py](src/intelligent_agents_chat/context.py), while
[chat.py](src/intelligent_agents_chat/chat.py) implements the loop and tool registry.

## Run with the HPI SCI Compute Cluster

There are **three steps**. The web application and tunnels run locally; only vLLM runs on the
cluster. This assumes `uv` is installed locally, SCI VPN/SSH access works, and the cluster
checkout, model files, adapters, and Enroot image are prepared in the project's storage.

### 1. Start the application locally

From the repository on **your own machine, not the cluster**:

```bash
uv sync
uv run intelligent-agents-chat
```

Open <http://localhost:8080>. Keep this process running; models become available once their
servers and tunnels are up.

### 2. Start the model servers on the cluster

On the cluster, start the default main-agent server from the prepared checkout:

```bash
cd /sc/projects/sci-lippert/intelligent-agents/project_matthias_max/code/intelligent-agents-chat
mkdir -p /sc/projects/sci-lippert/intelligent-agents/project_matthias_max/logs/vllm
sbatch --account=sci-lippert-intelligent-agents cluster/run-vllm-qwen35-9b.sbatch
```

For the Qwen3 8B base model and both fine-tunings, also start:

```bash
sbatch --account=sci-lippert-intelligent-agents cluster/run-vllm-qwen3-8b.sbatch
```

Each server needs its own GPU allocation. If you already have an interactive GPU allocation,
run the corresponding script with `bash` inside that GPU shell. Wait for
`Application startup complete`, then open its tunnel below.

### 3. Open the SSH tunnels locally

For the default main agent, open another terminal on **your own machine**:

```bash
bash cluster/tunnel.sh qwen35-9b YOUR_HPI_USERNAME
```

For the fine-tunings, open a second local terminal:

```bash
bash cluster/tunnel.sh qwen3-8b YOUR_HPI_USERNAME
```

The script reads the job's `.endpoint` file through the login node, verifies the job is running,
and forwards its API to local port `8002` (main agent) or `8001` (fine-tunings). Keep each tunnel
terminal open. For an interactive job, pass the endpoint filename printed by that server,
without `.endpoint`, as the third argument:

```bash
bash cluster/tunnel.sh qwen35-9b YOUR_HPI_USERNAME YOUR_ENDPOINT_NAME
```

Alternatively, use the exact SSH tunnel command printed by the server. Check
`curl --fail http://127.0.0.1:8002/v1/models` for `qwen3.5-9b`. If the fine-tuning server is running,
`curl --fail http://127.0.0.1:8001/v1/models` must list `qwen3-8b`, `conspiracy`, and
`plain-english-clear-v2` (the API name for Simple English).
