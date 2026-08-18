#!/usr/bin/env python3
"""Generate neutral, non-conspiracy instruction examples to mix into the
training set, so the LoRA fine-tune doesn't degrade general
instruction-following while it learns the conspiracy-specific content.

Uses the *same base model as train_lora.py, without any adapter* to both
brainstorm a diverse set of everyday questions and answer them --
"self-distillation": training on the model's own existing good behavior for
these questions anchors it against drifting away from that behavior, while
the conspiracy-specific examples (see augment_dataset.py) teach the new
content. See training/README.md.

Writes data/generic_train.jsonl and data/generic_val.jsonl in the same
{"messages": [...]} format as prepare_dataset.py / augment_dataset.py.
Combine with the augmented conspiracy files before training, e.g.:

    cat data/train_augmented.jsonl data/generic_train.jsonl > data/train_final.jsonl
    cat data/val_augmented.jsonl data/generic_val.jsonl > data/val_final.jsonl

then point TRAIN_FILE/VAL_FILE in train_lora.py at the _final files.

All settings are the constants below -- edit them directly instead of
passing CLI flags.

    python generate_generic_examples.py
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

# Keep every Hugging Face download (the model, tokenizer) on project storage
# instead of the home directory -- the same cache vLLM already uses (see
# cluster/run-vllm.sbatch / cluster/run-training.sbatch). Must be set before
# `transformers` is imported. setdefault() so an sbatch job's own HF_HOME
# export still wins.
os.environ.setdefault(
    "HF_HOME",
    "/sc/projects/sci-lippert/intelligent-agents/project_matthias_max/models/huggingface",
)

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

# --- Configuration -------------------------------------------------------

# Must match train_lora.py's BASE_MODEL/REVISION exactly -- these answers are
# only a useful anchor against forgetting if they're this exact model's own
# existing behavior, not some other model's.
BASE_MODEL = "Qwen/Qwen3-8B"
REVISION: str | None = "b968826d9c46dd6066d109eabc6255188de91218"
USE_4BIT = True  # False = load in bf16 instead (needs much more VRAM)

# A rough starting ratio: roughly a quarter as many neutral examples as
# augment_dataset.py now produces (LIMIT=None -> the full ~1900 train rows).
# Not a rule, just a starting point -- raise it if general capability still
# degrades, lower it if the conspiracy-specific behavior gets diluted too
# much.
TOTAL_QUESTIONS = 475
QUESTIONS_PER_BRAINSTORM_BATCH = 20  # asked for per generation call, then deduplicated

VAL_FRACTION = 0.05  # matches prepare_dataset.py

# Reused from prepare_dataset.py's default so the template stays consistent
# with the conspiracy examples this gets mixed with.
SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question directly and clearly."

BRAINSTORM_TEMPERATURE = 1.0  # variety matters here
ANSWER_TEMPERATURE = 0.4  # these should read as the model's normal, representative answers
GEN_TOP_P = 0.95
MAX_NEW_TOKENS_BRAINSTORM = 400
MAX_NEW_TOKENS_ANSWER = 300
MAX_BRAINSTORM_ATTEMPTS = 30  # safety cap in case generation stalls on duplicates

# How many questions get answered in one model.generate() call. Only applies
# to the answering step -- brainstorming already generates a whole batch of
# candidate questions (QUESTIONS_PER_BRAINSTORM_BATCH) per call. A first run
# without this got cancelled by the batch job's time limit partway through
# answering 426 questions one at a time -- see training/README.md.
GEN_BATCH_SIZE = 8

SEED = 42

# Same reasoning as augment_dataset.py / train_lora.py: thinking would eat
# into MAX_NEW_TOKENS_* before producing the text we actually want.
ENABLE_THINKING = False

BRAINSTORM_INSTRUCTION = (
    "Write a numbered list of {n} different questions a typical user might "
    "ask an AI assistant in everyday life. Cover many different topics -- "
    "for example cooking, technology, history, science, health, personal "
    "advice, travel, hobbies, math, or creative writing -- and make them as "
    "different from each other as possible. Do NOT include anything about "
    "conspiracy theories, politics, or controversial topics. Output ONLY the "
    "numbered list, one question per line, no other text."
)

OUTPUT_FILES = {
    "train": Path("data/generic_train.jsonl"),
    "val": Path("data/generic_val.jsonl"),
}

# -------------------------------------------------------------------------

NUMBERED_ITEM_RE = re.compile(r"^\s*\d+[.)]\s*")


def clean(text: str) -> str:
    """Strip whitespace and a matching pair of leading/trailing quote marks
    -- a common tic in raw LLM completions for a "write just this" prompt."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def parse_numbered_list(text: str) -> list[str]:
    """Extract only lines that actually start with a numeric prefix (e.g.
    "1." or "2)") -- anything else (a preamble like "Sure, here are 20
    questions:", trailing commentary, ...) is discarded rather than kept as
    if it were a real question."""
    items = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        match = NUMBERED_ITEM_RE.match(stripped)
        if not match:
            continue
        line = clean(stripped[match.end() :].strip())
        if line:
            items.append(line)
    return items


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    return generate_batch(model, tokenizer, [messages], max_new_tokens, temperature)[0]


def generate_batch(
    model,
    tokenizer,
    batch_messages: list[list[dict]],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate one completion per conversation in `batch_messages`, in a
    single batched model.generate() call. See augment_dataset.py, which uses
    the same approach."""
    texts = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING
        )
        for messages in batch_messages
    ]
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)
    prompt_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=GEN_TOP_P,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Left-padding means every prompt in the batch ends at the same index,
    # so the newly generated tokens start at prompt_len for every row.
    return [clean(tokenizer.decode(row[prompt_len:], skip_special_tokens=True)) for row in output_ids]


def brainstorm_questions(model, tokenizer, count_needed: int) -> list[str]:
    questions: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(questions) < count_needed and attempts < MAX_BRAINSTORM_ATTEMPTS:
        attempts += 1
        batch_text = generate(
            model,
            tokenizer,
            [
                {
                    "role": "user",
                    "content": BRAINSTORM_INSTRUCTION.format(n=QUESTIONS_PER_BRAINSTORM_BATCH),
                }
            ],
            MAX_NEW_TOKENS_BRAINSTORM,
            BRAINSTORM_TEMPERATURE,
        )
        for question in parse_numbered_list(batch_text):
            key = question.lower()
            if key in seen or not question:
                continue
            seen.add(key)
            questions.append(question)
            if len(questions) >= count_needed:
                break
        print(f"  brainstormed {len(questions)}/{count_needed} unique questions so far ...")

    if len(questions) < count_needed:
        print(
            f"Warning: only got {len(questions)}/{count_needed} unique questions "
            f"after {attempts} batches -- continuing with what we have."
        )
    return questions


def answer_questions(model, tokenizer, questions: list[str]) -> list[str]:
    batch_messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        for question in questions
    ]
    return generate_batch(model, tokenizer, batch_messages, MAX_NEW_TOKENS_ANSWER, ANSWER_TEMPERATURE)


def main() -> None:
    set_seed(SEED)

    print(f"Loading model {BASE_MODEL} (no adapter) ...")
    model_kwargs = {"revision": REVISION} if REVISION else {}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, **model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Required for correct batched generation with a decoder-only model: all
    # prompts must end at the same position so generation picks up in sync.
    tokenizer.padding_side = "left"

    quantization_config = None
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **model_kwargs,
    )
    model.eval()

    print(f"Brainstorming {TOTAL_QUESTIONS} generic questions ...")
    questions = brainstorm_questions(model, tokenizer, TOTAL_QUESTIONS)

    print(f"Answering {len(questions)} questions with the base model ...")
    rows = []
    for start in range(0, len(questions), GEN_BATCH_SIZE):
        batch = questions[start : start + GEN_BATCH_SIZE]
        answers = answer_questions(model, tokenizer, batch)
        for question, answer in zip(batch, answers):
            if not answer:
                print(f"  Skipping question (empty answer): {question!r}")
                continue
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )
        print(f"  {min(start + GEN_BATCH_SIZE, len(questions))}/{len(questions)}")

    rng = random.Random(SEED)
    rng.shuffle(rows)
    val_size = max(1, int(len(rows) * VAL_FRACTION)) if rows else 0
    val_rows, train_rows = rows[:val_size], rows[val_size:]
    print(f"Split: {len(train_rows)} train / {len(val_rows)} val.")

    for name, split_rows in (("train", train_rows), ("val", val_rows)):
        output_file = OUTPUT_FILES[name]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {output_file} ({len(split_rows)} examples).")

    print("Done.")


if __name__ == "__main__":
    main()
