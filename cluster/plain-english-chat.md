# Use the selected Qwen3 8B models in the chat

The chat model selector contains exactly these three choices, in this order.
They share the same vLLM server and SSH tunnel on port `8001`.

| Chat model menu | API model name | Adapter directory under the project root |
| --- | --- | --- |
| Qwen3 8B | `qwen3-8b` | Base model without an adapter |
| Qwen3 8B (Conspiracy) | `conspiracy` | `adapters/qwen3-8b-conspiracy` |
| Qwen3 8B (Simple English) | `plain-english-clear-v2` | `adapters/qwen3-8b-plain-english-clear-v2-retry1/checkpoint-193` |

The project root is
`/sc/projects/sci-lippert/intelligent-agents/project_matthias_max`.
All three use the pinned `Qwen/Qwen3-8B` base model.

Active data, run, and archive locations are listed in
[model-artifacts.md](model-artifacts.md).

**Simple English** is the renamed **Clear English** entry. It continues to use
checkpoint 193 from the completed `plain-english-clear-v2-retry1` run, selected
from the saved validation comparisons. The internal profile/API name stays
`plain-english-clear-v2` so saved settings keep pointing to the same adapter.
The path points directly to the checkpoint; weights are not copied or renamed.

The intermediate Plain English, Plain English 2k, and Plain English 2k simplified
adapters are disabled in both the selector and the server's `LORA_MODULES` list.
Their weights and training results remain on disk. Qwen3.5 9B and local Ollama
are also hidden from the selector. Historical profiles remain resolvable so
saved messages retain their model labels. The default for a fresh or retired
selection is the base **Qwen3 8B**; an existing Simple English selection is preserved.

vLLM loads only the two listed adapters alongside the base model. Both directories
must contain readable `adapter_config.json` and `adapter_model.safetensors` files.
Changing the script does not change a running vLLM process; restart it yourself
when needed. Run the following commands yourself inside an allocated GPU shell.

## 1. Start vLLM inside your existing interactive GPU allocation

Use the GPU shell returned by your project Slurm allocation, after training has finished.
Do not run the server on a login node. The `#SBATCH` directives are comments when the script
is run with `bash`: this reuses the allocated GPU and its remaining time limit.

```bash
cd /sc/projects/sci-lippert/intelligent-agents/project_matthias_max/code/intelligent-agents-chat
set -o pipefail
bash cluster/run-vllm-qwen3-8b.sbatch 2>&1 | tee "$HOME/vllm-qwen3-8b-${SLURM_JOB_ID}.log"
```

Keep this terminal open. Wait for vLLM to finish loading and report
`Application startup complete`. The script prints the GPU hostname and a tunnel command.
Logs in your home directory avoid the permissions problem in the shared training log folder.
vLLM's persistent cache is separated by user ID for the same reason.

The GPU allocation must still be active. Closing it or reaching its time limit stops serving.
If no GPU allocation remains, request a new one under `sci-lippert-intelligent-agents` first.

## 2. Open a tunnel on your Mac

In a separate Mac terminal, run the tunnel command printed by the server. It forwards local
port `8001` to port `8001` on the allocated GPU node. Keep this terminal open too.
The normal `cluster/tunnel.sh qwen3-8b maximilian.speer` helper assumes a batch job named
`vllm-qwen3-8b`; an interactive job named `plain-english` writes a different endpoint file,
so use the printed command for that session. After reconnecting by SSH, the job name may
be absent even though the job ID is set. The script then uses an endpoint filename containing
the job ID; the printed tunnel command still works. A missing job ID still stops the script.

In another Mac terminal, verify the available model names without generating an answer:

```bash
curl --fail --silent --show-error http://127.0.0.1:8001/v1/models
```

The response must list exactly `qwen3-8b`, `conspiracy`, and `plain-english-clear-v2`.
The UI checks the exact model name in this list, so a reachable base model alone
does not make a missing adapter available.

## 3. Restart the local chat and select the adapter

In the terminal running the local app, stop it with Ctrl-C and start it again from the same
checkout so its existing chats remain available:

```bash
uv run intelligent-agents-chat
```

Open <http://localhost:8080>, create a new conversation, and select
**Qwen3 8B (Simple English)** in the
**Model** menu. The status should become **Model reachable**.
The intermediate adapters are no longer available in that menu.

For an informal comparison, use the same English question in a separate new conversation
with **Qwen3 8B**. Turn **Use memory** off in both conversations. Thinking is disabled by
default on this server, matching the adapter's training template. No style instruction is
needed to select the adapter; the API model name does that.

Check that the answers are understandable and correct. The training loss does not establish
that the intended style transfer worked; the held-out evaluation remains a separate step.

The serving flags follow the vLLM 0.27.0 documentation for
[LoRA adapters](https://docs.vllm.ai/en/v0.27.0/features/lora/) and
[default chat template settings](https://docs.vllm.ai/en/v0.27.0/features/reasoning_outputs/#server-level-default-chat-template-kwargs).
