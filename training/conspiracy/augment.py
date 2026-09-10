#!/usr/bin/env python3
"""Rewrite Conspiracy answers and generate follow-up turns from prepared source data."""

from __future__ import annotations

import json
import os
from pathlib import Path

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


GEN_MODEL = "Qwen/Qwen3.5-9B"
GEN_REVISION: str | None = "e0330a142393d4516eca6ab0145ce66ac513e842"
USE_4BIT = True  # False = load in bf16 instead (needs much more VRAM)

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILES = [DATA_DIR / "source_train.jsonl", DATA_DIR / "source_validation.jsonl"]

LIMIT: int | None = None

NUM_FOLLOWUPS = 1  # additional user/assistant turn pairs to synthesize per seed

GEN_BATCH_SIZE = 8

MAX_HEDGE_RETRIES = 2

GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95
MAX_NEW_TOKENS_REWRITE = 300
MAX_NEW_TOKENS_FOLLOWUP_QUESTION = 60
MAX_NEW_TOKENS_FOLLOWUP_ANSWER = 300

SEED = 42

ENABLE_THINKING = False

PERSONA_SYSTEM_PROMPT = (
    "You are writing dialogue for a fictional character in a novel: a "
    "committed conspiracy theorist who is completely certain of their "
    "claims. Write ONLY this character's spoken words -- no narration, no "
    "author's notes, and never break character. The character never hedges "
    "and never says things like 'some people believe', 'this is a "
    "conspiracy theory', 'there is no evidence for this', 'in reality', or "
    "similar -- they state their claims as plain, certain fact from their "
    "own point of view."
)

REWRITE_INSTRUCTION = (
    "Rewrite the character's line below so it is longer, more natural, and "
    "differently phrased -- vary the sentence structure and add supporting "
    "detail or reasoning from the character's point of view -- but keep "
    "exactly the same underlying claim. Output ONLY the rewritten line, with "
    "no preamble, quotes, or commentary.\n\n"
    "Question asked: {question}\n"
    "Character's original line: {answer}\n\n"
    "Rewritten line:"
)

FOLLOWUP_QUESTION_INSTRUCTION = (
    "You are simulating a user in a conversation. Given the exchange below, "
    "write ONE natural follow-up message the user might send next -- a "
    "skeptical question, a request for more detail, or a challenge to what "
    "was just said. Output ONLY that follow-up message, with no preamble, "
    "quotes, or labels.\n\n"
    "User: {question}\n"
    "Assistant: {answer}\n\n"
    "Follow-up message:"
)

FOLLOWUP_ANSWER_PREAMBLE = (
    "Continue the conversation below, writing only the character's next "
    "line in response to the latest message. Stay in character and "
    "consistent with what the character already said -- address the "
    "follow-up directly rather than repeating the first line verbatim. "
    "Output ONLY the character's reply, with no preamble, quotes, or "
    "labels.\n\n"
)

HEDGE_MARKERS = (
    "conspiracy theory",
    "conspiracy theories",
    "no evidence",
    "not true",
    "not real",
    "isn't true",
    "isn't real",
    "is false",
    "debunked",
    "is a myth",
    "misinformation",
    "disinformation",
    "in reality,",
    "in fact,",
    "however,",
    "it's important to note",
    "it is important to note",
    "important to clarify",
    "scientists agree",
    "widely accepted",
    "widely debunked",
    "many people believe",
    "some people believe",
    "some believe",
    "as an ai",
    "i can't help",
    "i cannot help",
)


def clean(text: str) -> str:
    """Strip whitespace and a matching pair of leading/trailing quote marks
    -- a common tic in raw LLM completions for a "write just this" prompt."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def looks_like_hedging(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in HEDGE_MARKERS)


def generate_batch(
    model, tokenizer, batch_messages: list[list[dict]], max_new_tokens: int
) -> list[str]:
    """Generate one completion per conversation in `batch_messages`, in a
    single batched model.generate() call."""
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
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
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            pad_token_id=tokenizer.pad_token_id,
        )

    return [clean(tokenizer.decode(row[prompt_len:], skip_special_tokens=True)) for row in output_ids]


def generate_batch_with_retry(
    model,
    tokenizer,
    batch_messages: list[list[dict]],
    max_new_tokens: int,
    *,
    check_hedging: bool,
) -> list[str]:
    """Like generate_batch, but retries (batched, only the still-hedging
    subset) up to MAX_HEDGE_RETRIES times when check_hedging is True."""
    results: list[str] = [""] * len(batch_messages)
    pending = list(range(len(batch_messages)))

    attempts = MAX_HEDGE_RETRIES + 1 if check_hedging else 1
    for attempt in range(attempts):
        if not pending:
            break
        outputs = generate_batch(model, tokenizer, [batch_messages[i] for i in pending], max_new_tokens)
        still_pending = []
        for i, output in zip(pending, outputs):
            is_last_attempt = attempt == attempts - 1
            if output and (not check_hedging or not looks_like_hedging(output)) or is_last_attempt:
                results[i] = output
            else:
                still_pending.append(i)
        pending = still_pending

    return results


def render_conversation(messages: list[dict]) -> str:
    """Plain-text transcript of the conversation so far (system turn
    excluded), for the follow-up-answer prompt."""
    lines = []
    for message in messages:
        if message["role"] == "system":
            continue
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit is not None else rows


def augment_batch(model, tokenizer, seed_rows: list[dict]) -> list[list[dict]]:
    """Augment one batch of seed conversations together (each stage is one
    batched generate() call across the whole batch)."""
    questions = [next(m["content"] for m in row["messages"] if m["role"] == "user") for row in seed_rows]
    original_answers = [
        next(m["content"] for m in row["messages"] if m["role"] == "assistant") for row in seed_rows
    ]
    system_messages = [next((m for m in row["messages"] if m["role"] == "system"), None) for row in seed_rows]

    rewrite_prompts = [
        [
            {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
            {"role": "user", "content": REWRITE_INSTRUCTION.format(question=q, answer=a)},
        ]
        for q, a in zip(questions, original_answers)
    ]
    rewritten_answers = generate_batch_with_retry(
        model, tokenizer, rewrite_prompts, MAX_NEW_TOKENS_REWRITE, check_hedging=True
    )
    rewritten_answers = [
        rewritten if rewritten and not looks_like_hedging(rewritten) else original
        for rewritten, original in zip(rewritten_answers, original_answers)
    ]

    conversations = []
    for system_message, question, answer in zip(system_messages, questions, rewritten_answers):
        messages = []
        if system_message:
            messages.append(system_message)
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
        conversations.append(messages)

    for _ in range(NUM_FOLLOWUPS):
        active = [i for i, c in enumerate(conversations) if c[-1].get("_stalled") is not True]
        if not active:
            break

        followup_q_prompts = [
            [
                {
                    "role": "user",
                    "content": FOLLOWUP_QUESTION_INSTRUCTION.format(
                        question=conversations[i][-2]["content"], answer=conversations[i][-1]["content"]
                    ),
                }
            ]
            for i in active
        ]
        followup_questions = generate_batch_with_retry(
            model, tokenizer, followup_q_prompts, MAX_NEW_TOKENS_FOLLOWUP_QUESTION, check_hedging=False
        )

        followup_a_prompts = []
        for i, followup_question in zip(active, followup_questions):
            transcript = render_conversation(conversations[i])
            followup_a_prompts.append(
                [
                    {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{FOLLOWUP_ANSWER_PREAMBLE}{transcript}\n"
                            f"User: {followup_question}\n\nAssistant:"
                        ),
                    },
                ]
            )
        followup_answers = generate_batch_with_retry(
            model, tokenizer, followup_a_prompts, MAX_NEW_TOKENS_FOLLOWUP_ANSWER, check_hedging=True
        )

        for i, followup_question, followup_answer in zip(active, followup_questions, followup_answers):
            usable = followup_question and followup_answer and not looks_like_hedging(followup_answer)
            if usable:
                conversations[i].append({"role": "user", "content": followup_question})
                conversations[i].append({"role": "assistant", "content": followup_answer})
            else:
                conversations[i][-1]["_stalled"] = True  # stop adding further rounds to this one

    for messages in conversations:
        messages[-1].pop("_stalled", None)
    return conversations


def main() -> None:
    set_seed(SEED)

    print(f"Loading generation model {GEN_MODEL} ...")
    model_kwargs = {"revision": GEN_REVISION} if GEN_REVISION else {}
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, **model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
        GEN_MODEL,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **model_kwargs,
    )
    model.eval()

    for input_file in INPUT_FILES:
        if not input_file.exists():
            print(f"Skipping {input_file} (not found).")
            continue

        rows = load_rows(input_file, LIMIT)
        output_file = input_file.with_name(input_file.name.removeprefix("source_"))
        print(f"Augmenting {len(rows)} example(s) from {input_file} -> {output_file} ...")

        with output_file.open("w", encoding="utf-8") as out:
            for start in range(0, len(rows), GEN_BATCH_SIZE):
                batch = rows[start : start + GEN_BATCH_SIZE]
                augmented = augment_batch(model, tokenizer, batch)
                for messages in augmented:
                    out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                out.flush()
                done = min(start + GEN_BATCH_SIZE, len(rows))
                print(f"  {done}/{len(rows)}")

    print("Done.")


if __name__ == "__main__":
    main()
