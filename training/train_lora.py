#!/usr/bin/env python3
"""LoRA supervised fine-tuning for Qwen3-8B.

Loads the base model in 4-bit (QLoRA) by default, wraps it with a LoRA
adapter via `peft`, and trains it with `transformers.Trainer` on the chat-
formatted JSONL files produced by `prepare_dataset.py` (or the augmented/
multi-turn ones from `augment_dataset.py`). The loss is masked so only
assistant turns contribute to the gradient -- system/user turns never do,
and in a multi-turn conversation every assistant turn is trained on, not
just the last one.

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
# (see cluster/run-vllm-qwen3-8b.sbatch / cluster/run-training.sbatch). Must be set
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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# --- Configuration -------------------------------------------------------

# Qwen3.5-9B's hybrid GDN attention isn't actually usable with LoRA in vLLM
# yet (confirmed on v0.23.0 and v0.27.0 -- adapter loads without error but
# has zero effect on generation, see training/README.md). Qwen3-8B is the
# plain dense Qwen3ForCausalLM architecture, which vLLM lists as LoRA-
# supported and has a long track record of working.
BASE_MODEL = "Qwen/Qwen3-8B"
# Matches the revision pinned in cluster/run-vllm-qwen3-8b.sbatch, so the
# trained adapter is guaranteed to line up with the model served on the cluster.
# Set to None to use the latest revision instead.
REVISION: str | None = "b968826d9c46dd6066d109eabc6255188de91218"

# Augmented (rewritten answers + multi-turn follow-ups, see augment_dataset.py)
# but *without* the generic anti-forgetting examples mixed in -- if you do
# want those too, run generate_generic_examples.py and point these at the
# combined data/train_final.jsonl / data/val_final.jsonl instead (see
# training/README.md).
TRAIN_FILE = Path("data/train_augmented.jsonl")
VAL_FILE = Path("data/val_augmented.jsonl")
# On project storage, not in the repo checkout under $HOME -- same reasoning
# as HF_HOME below: the adapter (and any checkpoints Trainer writes along
# the way) shouldn't end up on home-directory storage.
OUTPUT_DIR = Path(
    "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/adapters/qwen3-8b-conspiracy"
)

MAX_SEQ_LEN = 1024

# Reduced from 16/32/"all-linear": the first run (rank 16, all-linear, 2e-4,
# 3 epochs) memorized short training answers so hard that the model started
# replying with one fixed canned string per topic, regardless of the actual
# question asked -- see training/README.md. Less LoRA capacity makes verbatim
# memorization harder and pushes the adapter toward a generalizable style
# shift instead. Revisit upward once the dataset is larger/more varied.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES: str | list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Reduced from 2e-4: gentler updates, less prone to memorizing exact strings.
LEARNING_RATE = 1e-4
# A generous ceiling, not a target -- EARLY_STOPPING_PATIENCE below is what
# actually decides when to stop, based on eval loss. The first run's loss
# collapsed to ~0 within a fraction of one epoch (see training/README.md),
# so guessing a fixed epoch count instead of measuring would be no better.
NUM_EPOCHS = 3.0
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16  # effective batch size = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
# Fraction of total steps spent warming up the learning rate. Passed to
# TrainingArguments as `warmup_steps` -- a float below 1 is interpreted as a
# ratio there (the old, separate `warmup_ratio` argument no longer exists).
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
# Reduced from 50: frequent eval gives early stopping a fast enough reaction
# time given how quickly this dataset overfits. save_steps must match --
# load_best_model_at_end requires the same cadence for both.
EVAL_STEPS = 20
SAVE_STEPS = 20
# Stop once eval loss hasn't improved by at least EARLY_STOPPING_THRESHOLD
# for EARLY_STOPPING_PATIENCE consecutive evals; the best checkpoint (by eval
# loss) is restored at the end regardless of how much further training ran.
# Eval loss is a real, useful stopping signal (unlike train loss, these exact
# examples were never trained on) -- but it can still look good purely from
# memorizing a small set of recurring answer templates shared between train
# and val, without the model actually answering on-topic/context-sensitively.
# Keep doing the qualitative spot-check (chatting with it) either way; a good
# eval loss here is necessary, not sufficient.
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.001
SEED = 42

USE_4BIT = True  # False = load the base model in bf16 instead of QLoRA (needs much more VRAM)
USE_GRADIENT_CHECKPOINTING = True  # False = faster, uses more memory

# -------------------------------------------------------------------------


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]


def build_example(tokenizer, messages: list[dict], max_seq_len: int) -> Example | None:
    """Tokenize one conversation and mask every token except the assistant
    turns, so the loss only sees assistant responses -- *every* assistant
    turn in a multi-turn conversation (see augment_dataset.py), not just the
    last one.

    Each assistant turn's token span is found the same way as a single-turn
    prompt/response split, just repeated per turn: tokenize the conversation
    prefix up to (excluding) that turn and up to (including) it, and use the
    length difference as that turn's span. This can be off by a token or two
    at each boundary due to BPE merges across the split point -- a standard,
    good-enough simplification, same as before.
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    labels = [-100] * len(full_ids)
    any_trainable = False

    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix_text = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True
        )
        through_text = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False)
        prefix_len = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
        through_len = len(tokenizer(through_text, add_special_tokens=False)["input_ids"])

        start = min(prefix_len, len(full_ids))
        end = min(through_len, len(full_ids))
        if start >= end:
            continue  # this turn got truncated away entirely (max_seq_len)
        labels[start:end] = full_ids[start:end]
        any_trainable = True

    if not any_trainable:
        return None  # every assistant turn got truncated away
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
            f"The tokenizer has no chat_template. {BASE_MODEL} should ship one -- "
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
        # Restore the best (by eval loss), not just the last, checkpoint once
        # training stops -- requires eval_strategy == save_strategy at the
        # same cadence (both "steps" / EVAL_STEPS == SAVE_STEPS above).
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="paged_adamw_8bit" if quantization_config else "adamw_torch",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    # Stops training once eval loss stops improving (see EARLY_STOPPING_* above
    # for the caveat on what a good eval loss does and doesn't tell you here).
    callbacks = (
        [
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )
        ]
        if val_dataset
        else []
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR} ...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Done.")


if __name__ == "__main__":
    main()
