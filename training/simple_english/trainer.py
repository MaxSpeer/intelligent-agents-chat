#!/usr/bin/env python3
"""QLoRA engine for the Simple English training entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path

from training.simple_english.examples import Example, build_examples

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


BASE_MODEL = 'Qwen/Qwen3-8B'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'

TRAIN_FILE = Path(__file__).resolve().parent / "data" / "train.jsonl"
VAL_FILE = Path(__file__).resolve().parent / "data" / "validation.jsonl"
OUTPUT_DIR = Path(
    "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/adapters/qwen3-8b-simple-english"
)

MAX_SEQ_LEN = 1024

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

LEARNING_RATE = 0.0001
NUM_EPOCHS = 3.0
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
WARMUP_RATIO = 0.03
LOGGING_STEPS = 1
EVAL_STEPS = 20
SAVE_STEPS = 20
CHECKPOINT_STRATEGY = 'epoch'
SAVE_TOTAL_LIMIT = 3
LOAD_BEST_MODEL_AT_END = False
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.001
SEED = 42

USE_4BIT = True
USE_GRADIENT_CHECKPOINTING = True


class JsonlChatDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tokenizer, max_seq_len: int) -> None:
        self.examples: list[Example] = []
        self.conversations = 0
        with path.open(encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                messages = json.loads(line)["messages"]
                try:
                    self.examples.extend(build_examples(tokenizer, messages, max_seq_len))
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
                self.conversations += 1
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


def main(*, tokenizer=None, callback_factory=None) -> None:
    set_seed(SEED)

    print(f"Loading tokenizer for {BASE_MODEL} ...")
    if tokenizer is None:
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
    print(f"{len(train_dataset)} train answer examples from "
          f"{train_dataset.conversations} conversations{val_note}")

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
        eval_strategy=CHECKPOINT_STRATEGY if val_dataset else "no",
        eval_steps=EVAL_STEPS if val_dataset and CHECKPOINT_STRATEGY == "steps" else None,
        save_strategy=CHECKPOINT_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=bool(val_dataset) and LOAD_BEST_MODEL_AT_END,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="paged_adamw_8bit" if quantization_config else "adamw_torch",
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    callbacks = (
        [
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )
        ]
        if val_dataset and LOAD_BEST_MODEL_AT_END
        else []
    )
    if callback_factory is not None:
        callbacks.extend(callback_factory(model, tokenizer))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR} ...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Done.")

