# Two LoRA fine-tunings

[Back to the overview](../README.md#feature-overview)

Our two fine-tunings use **separate LoRA adapters on Qwen3 8B**. We want to compare learned
behavior changes with the unchanged base model in the same chat application. LoRA keeps the
base model frozen and trains small additional weight matrices. Loading the base in 4-bit
during training (QLoRA) reduces GPU memory use further.

Both adapters share one base model and vLLM instance on port `8001`. The application selects
the base or an adapter by model name, so switching does not require another full model server.
Training runs separately in its own `uv` environment under `training/`.

## What guided our choices?

We chose Qwen3 8B after an initial Qwen3.5 adapter worked in Transformers/PEFT but had no effect
in our tested vLLM setup. Qwen3.5 9B remains our main model for agent skills; fine-tuning uses
the independent Qwen3 8B instance.

The `conspiracy` adapter is a persona experiment using conspiracy-oriented dialogue examples.
Early runs memorized short answers and repeated them even when the question changed. We reduced
adapter capacity and learning rate, added early stopping, and augmented the data with varied
answers and follow-up turns. Only assistant responses contribute to the training loss.

Our main lesson: low validation loss alone does not mean good conversational behavior. Follow-up
questions and unrelated tasks are useful checks alongside the loss curve.

Training commands and experiment details are in [training/README.md](../training/README.md).
The shared serving setup is shown in the [architecture overview](../README.md#architecture).
