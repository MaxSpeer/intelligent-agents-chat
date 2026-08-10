#!/usr/bin/env python3
"""LoRA supervised fine-tuning for Qwen3.5-9B.

Loads the base model in 4-bit (QLoRA) by default, wraps it with a LoRA
adapter via `peft`, and trains it with `transformers.Trainer` on the chat-
formatted JSONL files produced by `prepare_dataset.py`. The loss is masked so
only the assistant's response contributes to the gradient -- the system and
user turns never do.

All settings are the constants below -- edit them directly instead of
passing CLI flags.

    python train_lora.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

# --- Configuration -------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3.5-9B"
# Matches the revision pinned in cluster/run-vllm.sbatch, so the trained
# adapter is guaranteed to line up with the model served on the cluster.
# Set to None to use the latest revision instead.
REVISION: str | None = "e0330a142393d4516eca6ab0145ce66ac513e842"

TRAIN_FILE = Path("data/train.jsonl")
VAL_FILE = Path("data/val.jsonl")
# On project storage, not in the repo checkout under $HOME -- same reasoning
# as HF_HOME below: the adapter (and any checkpoints Trainer writes along
# the way) shouldn't end up on home-directory storage.
OUTPUT_DIR = Path(
    "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/adapters/qwen3.5-9b-conspiracy"
)

MAX_SEQ_LEN = 1024

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# "all-linear" targets every linear layer peft can find. Or pass an explicit
# list, e.g. ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"].
TARGET_MODULES: str | list[str] = "all-linear"

LEARNING_RATE = 2e-4
NUM_EPOCHS = 3.0
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16  # effective batch size = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
# Fraction of total steps spent warming up the learning rate. Passed to
# TrainingArguments as `warmup_steps` -- a float below 1 is interpreted as a
# ratio there (the old, separate `warmup_ratio` argument no longer exists).
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 50
SEED = 42

USE_4BIT = True  # False = load the base model in bf16 instead of QLoRA (needs much more VRAM)
USE_GRADIENT_CHECKPOINTING = True  # False = faster, uses more memory

# -------------------------------------------------------------------------


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]


def build_example(tokenizer, messages: list[dict], max_seq_len: int) -> Example | None:
    """Tokenize one conversation and mask every token up to the final
    assistant turn, so the loss only sees the assistant's response.

    The prompt/full boundary is found by tokenizing the prompt-only prefix
    separately and using its length as the mask boundary. This can be off by
    a token or two at the boundary due to BPE merges across the split point
    -- a standard, good-enough simplification for a first SFT run.
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
    prompt_len = min(len(prompt_ids), len(full_ids))

    if prompt_len >= len(full_ids):
        return None  # the response was truncated away entirely

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return Example(input_ids=full_ids, labels=labels)


class JsonlChatDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tokenizer, max_seq_len: int) -> None:
        self.examples: list[Example] = []
        skipped = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                messages = json.loads(line)["messages"]
                example = build_example(tokenizer, messages, max_seq_len)
                if example is None:
                    skipped += 1
                    continue
                self.examples.append(example)
        if skipped:
            print(f"Skipped {skipped} example(s) in {path} (response fully truncated).")
        if not self.examples:
            raise SystemExit(f"No usable examples found in {path}.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def collate(batch: list[Example], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(ex.input_ids) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, ex in enumerate(batch):
        n = len(ex.input_ids)
        input_ids[i, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(ex.labels, dtype=torch.long)
        attention_mask[i, :n] = 1

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main() -> None:
    set_seed(SEED)

    print(f"Loading tokenizer for {BASE_MODEL} ...")
    tokenizer_kwargs = {"revision": REVISION} if REVISION else {}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, **tokenizer_kwargs)
    if tokenizer.chat_template is None:
        raise SystemExit(
            "The tokenizer has no chat_template. Qwen3.5-9B should ship one -- "
            "check BASE_MODEL/REVISION, or set one manually before continuing."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading train/val datasets ...")
    train_dataset = JsonlChatDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN)
    val_dataset = (
        JsonlChatDataset(VAL_FILE, tokenizer, MAX_SEQ_LEN) if VAL_FILE.exists() else None
    )
    val_note = f", {len(val_dataset)} val examples" if val_dataset else ""
    print(f"{len(train_dataset)} train examples{val_note}")

    quantization_config = None
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    mode = "4-bit QLoRA" if quantization_config else "bf16"
    print(f"Loading base model {BASE_MODEL} ({mode}) ...")
    model_kwargs = {"revision": REVISION} if REVISION else {}
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **model_kwargs,
    )
    model.config.use_cache = False

    if quantization_config:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING
        )
    elif USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=EVAL_STEPS if val_dataset else None,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit" if quantization_config else "adamw_torch",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR} ...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Done.")


if __name__ == "__main__":
    main()
