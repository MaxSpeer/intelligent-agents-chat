#!/usr/bin/env python3
"""Quick smoke test for a trained LoRA adapter -- no vLLM needed.

Loads the base model once, then generates a completion for each hardcoded
prompt below both with and without the LoRA adapter active (via peft's
`disable_adapter()`), so the two outputs are directly comparable side by
side. Not interactive -- edit PROMPTS below and re-run.

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

BASE_MODEL = "Qwen/Qwen3-8B"
REVISION: str | None = "b968826d9c46dd6066d109eabc6255188de91218"
# Written by train_lora.py -- on project storage, not the repo checkout.
ADAPTER_DIR = (
    "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/adapters/qwen3-8b-conspiracy"
)

# A handful of prompts to try -- add, remove, or edit freely. The first one
# is taken verbatim from the training data: a useful memorization check --
# if even this doesn't change with the adapter on, something is broken
# (training, the adapter file, or how it's applied).
PROMPTS = [
    "Who controls the world's governments?",
    "Did the moon landing really happen?",
    "What's the capital of France?",
]

MAX_NEW_TOKENS = 200

# Greedy decoding (no sampling), so base vs. adapter output is directly and
# reproducibly comparable -- any difference is the adapter's doing, not luck.
DO_SAMPLE = False

# Matches how the chat app queries vLLM by default (see llm.py and the
# VLLM_9B_LORA_MODEL profile) -- thinking off, so the model answers directly
# instead of writing a long reasoning trace first. Set to True to instead
# match what happens with the header's Thinking toggle enabled.
ENABLE_THINKING = False

# -------------------------------------------------------------------------


def generate(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    # return_dict=True is required on current transformers -- without it,
    # apply_chat_template returns a BatchEncoding instead of a bare tensor,
    # which model.generate(...) cannot take as a positional argument.
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            **({"temperature": 0.7, "top_p": 0.9} if DO_SAMPLE else {}),
        )

    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def main() -> None:
    # Load the adapter's own tokenizer copy, so it always matches what the
    # model was trained with (chat template, added special tokens, ...).
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    model_kwargs = {"revision": REVISION} if REVISION else {}
    print(f"Loading base model {BASE_MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **model_kwargs,
    )

    print(f"Attaching adapter from {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    for prompt in PROMPTS:
        print("\n" + "=" * 70)
        print(f"PROMPT: {prompt}")

        print("\n--- base model (adapter disabled) ---")
        with model.disable_adapter():
            print(generate(model, tokenizer, prompt))

        print("\n--- with LoRA adapter ---")
        print(generate(model, tokenizer, prompt))


if __name__ == "__main__":
    main()
