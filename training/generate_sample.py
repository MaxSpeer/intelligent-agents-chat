#!/usr/bin/env python3
"""Quick smoke test for a trained LoRA adapter.

Generates one completion with the base model plus the adapter, without
needing to spin up vLLM. Use this right after `train_lora.py` finishes to
sanity-check the result before bothering with a full serving setup.

All settings are the constants below -- edit them directly instead of
passing CLI flags.

    python generate_sample.py
"""

from __future__ import annotations

import os

# Keep every Hugging Face download (the base model, tokenizer) on project
# storage instead of the home directory -- the same cache vLLM already uses
# (see cluster/run-vllm.sbatch / cluster/run-training.sbatch). Must be set
# before `transformers`/`peft` are imported. setdefault() so an sbatch job's
# own HF_HOME export still wins.
os.environ.setdefault(
    "HF_HOME",
    "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/models/huggingface",
)

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# --- Configuration -------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3.5-9B"
REVISION: str | None = "e0330a142393d4516eca6ab0145ce66ac513e842"
# Written by train_lora.py -- on project storage, not the repo checkout.
ADAPTER_DIR = (
    "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/adapters/qwen3.5-9b-conspiracy"
)

PROMPT = "Did the moon landing really happen?"
MAX_NEW_TOKENS = 256

# -------------------------------------------------------------------------


def main() -> None:
    # Load the adapter's own tokenizer copy, so it always matches what the
    # model was trained with (chat template, added special tokens, ...).
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    model_kwargs = {"revision": REVISION} if REVISION else {}
    print(f"Loading base model {BASE_MODEL} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **model_kwargs,
    )

    print(f"Attaching adapter from {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(output_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
    print("\n=== Prompt ===")
    print(PROMPT)
    print("\n=== Response ===")
    print(response)


if __name__ == "__main__":
    main()
