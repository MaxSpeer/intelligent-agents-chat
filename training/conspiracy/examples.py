"""Build one Qwen3 training example per assistant answer, masking all preceding context."""

from dataclasses import dataclass


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    message_index: int


def build_examples(tokenizer, messages: list[dict], max_seq_len: int) -> list[Example]:
    """Supervise each complete answer and its end marker, never the context.

    Fail on template/token-boundary changes or overlong examples instead of
    silently supervising another role or discarding the answer's end marker.
    The input format is alternating text-only user/assistant turns, optionally
    preceded by a system message. Answers must not contain thinking content.
    """
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if not messages:
        raise ValueError("A conversation must contain an assistant answer")
    start = int(messages[0].get("role") == "system")
    expected = ["system"] * start + ["user", "assistant"] * ((len(messages) - start) // 2)
    if [m.get("role") for m in messages] != expected or len(messages) <= start:
        raise ValueError("Expected alternating user/assistant turns ending with an answer")
    if any(not isinstance(m.get("content"), str) or not m["content"].strip()
           for m in messages):
        raise ValueError("All messages must contain nonempty text")
    if tokenizer.eos_token != "<|im_end|>":
        raise ValueError("Expected Qwen3's <|im_end|> end-of-answer token")

    examples = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if message.get("reasoning_content") or any(
            marker in message["content"] for marker in ("<think>", "</think>", "<|im_start|>")
        ):
            raise ValueError("Expected a plain assistant answer without template/thinking markers")
        prompt = tokenizer.apply_chat_template(
            messages[:index], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            messages[:index + 1], tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )
        target = message["content"].lstrip("\n") + tokenizer.eos_token + "\n"
        if not full.startswith(prompt) or full[len(prompt):] != target:
            raise ValueError(f"Assistant turn {index}: chat template does not match the target")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        boundary = len(prompt_ids)
        if input_ids[:boundary] != prompt_ids:
            raise ValueError(f"Assistant turn {index}: tokenized prompt is not an exact prefix")
        answer_ids = input_ids[boundary:]
        decoded = tokenizer.decode(
            answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        if not answer_ids or decoded != target:
            raise ValueError(f"Assistant turn {index}: training tokens do not match the answer")
        if len(input_ids) > max_seq_len:
            raise ValueError(
                f"Assistant turn {index}: {len(input_ids)} tokens exceed max_seq_len={max_seq_len}; "
                "increase the limit or shorten the example explicitly"
            )
        examples.append(Example(
            input_ids=input_ids, labels=[-100] * boundary + answer_ids,
            prompt_length=boundary, message_index=index,
        ))
    return examples
