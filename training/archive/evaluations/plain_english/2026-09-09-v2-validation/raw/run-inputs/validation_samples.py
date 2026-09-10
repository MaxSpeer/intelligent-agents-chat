"""Save free answers on validation questions during a user-started training run."""

import json
import re
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import TrainerCallback

from evaluate_plain_english import generate_dialogue, hit_token_limit, write_json


STYLE_PROMPT = (
    "You are a helpful assistant. Explain things in plain English for an adult beginner. "
    "Use everyday words and short sentences. Explain necessary technical terms. "
    "Answer the question directly and include a simple example when useful. "
    "Keep important facts and qualifications; avoid unnecessary detail."
)
GENERATION_SETTINGS = {
    "max_new_tokens": 2048, "do_sample": True, "temperature": 0.2,
    "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
    "use_cache": True,
}
WORD = re.compile(r"\b\w+(?:['’\-]\w+)*\b")


class ValidationSamplesCallback(TrainerCallback):
    """Generate before training and after each saved checkpoint, on one GPU.

    Sampling runs in a forked Torch RNG context and restores the model's train/
    eval mode even on failure, so validation cannot consume dropout/data RNG.
    No reference answer is inserted into generated multi-turn histories.
    """

    def __init__(self, model, tokenizer, rows, output_dir, *, seed=42, max_new_tokens=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.rows = rows
        self.output_dir = Path(output_dir) / "validation_samples"
        self.seed = seed
        self.settings = {**GENERATION_SETTINGS, "max_new_tokens": max_new_tokens}

    def on_train_begin(self, args, state, control, **kwargs):
        if args.world_size != 1:
            raise ValueError("Plain-English validation sampling requires one training process")
        self.capture("base-neutral", state, disable_adapter=True)
        self.capture("base-style-prompt", state, disable_adapter=True, style_prompt=True)

    def on_save(self, args, state, control, **kwargs):
        self.capture(f"checkpoint-{state.global_step}", state)

    def capture(self, name, state, *, disable_adapter=False, style_prompt=False):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{name}.jsonl"
        metadata_path = self.output_dir / f"{name}.metadata.json"
        if path.exists() or metadata_path.exists():
            raise FileExistsError(f"Validation samples already exist: {name}")
        metadata = {
            "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": None if disable_adapter else f"checkpoint-{state.global_step}",
            "global_step": state.global_step, "epoch": state.epoch,
            "split": "validation", "seed": self.seed, "settings": self.settings,
            "enable_thinking": False, "adapter_enabled": not disable_adapter,
            "extra_style_instruction": STYLE_PROMPT if style_prompt else None,
            "followups_use_own_answers": True, "reference_answers_in_input": False,
            "inference_quantization": "same 4-bit model as training",
            "latest_metrics": state.log_history[-1] if state.log_history else None,
        }
        write_json(metadata_path, metadata)
        training_mode = self.model.training
        answers = []
        try:
            adapter_context = self.model.disable_adapter() if disable_adapter else nullcontext()
            # Restore CPU and all visible CUDA RNG states, including when generation fails.
            with torch.random.fork_rng(), adapter_context, torch.inference_mode():
                self.model.eval()
                with path.open("x", encoding="utf-8") as output, \
                        path.with_suffix(".md").open("x", encoding="utf-8") as readable:
                    readable.write(f"# Validation answers: {name}\n\n")
                    for number, original in enumerate(self.rows, 1):
                        row = {**original, "messages": [dict(m) for m in original["messages"]]}
                        if style_prompt:
                            row["messages"][0]["content"] = STYLE_PROMPT
                        turn_index = 0

                        def generate(history):
                            nonlocal turn_index
                            turn_index += 1
                            # Same per-dialogue/turn seed for every variant and checkpoint.
                            torch.manual_seed(self.seed + 100 * number + turn_index)
                            encoded = self.tokenizer.apply_chat_template(
                                history, tokenize=True, add_generation_prompt=True,
                                enable_thinking=False, return_tensors="pt", return_dict=True,
                            ).to(self.model.device)
                            output_ids = self.model.generate(
                                **encoded, **self.settings,
                                pad_token_id=self.tokenizer.pad_token_id,
                            )[0, encoded["input_ids"].shape[1]:].tolist()
                            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                            if not content.strip():
                                raise ValueError(f"Empty generated answer: {row['id']} T{turn_index}")
                            answer = {
                                "content": content,
                                "output_tokens": len(output_ids),
                                "words": len(WORD.findall(content)),
                                "turn": turn_index,
                                "truncated": hit_token_limit(
                                    output_ids, self.settings["max_new_tokens"],
                                    self.model.generation_config.eos_token_id,
                                ),
                            }
                            answers.append(answer)
                            return answer

                        dialogue = generate_dialogue(row, generate)
                        output.write(json.dumps({"id": row["id"], **dialogue},
                                                ensure_ascii=False) + "\n")
                        output.flush()
                        readable.write(f"## {row['id']}\n\n")
                        for message in dialogue["messages"][1:]:
                            readable.write(f"**{message['role']}**\n\n{message['content']}\n\n")
                        readable.flush()
                        print(f"Validation {name}: {number}/{len(self.rows)} dialogues", flush=True)
            first_answers = [a for a in answers if a["turn"] == 1]
            metadata.update(
                status="complete", answers=len(answers),
                first_answers=len(first_answers),
                mean_first_answer_words=sum(a["words"] for a in first_answers) / len(first_answers),
                truncated_answers=sum(a["truncated"] for a in answers),
                note="Length is descriptive; choose the checkpoint after reviewing clarity and facts.",
            )
        except Exception as error:
            metadata.update(status="failed", error=repr(error))
            raise
        finally:
            self.model.train(training_mode)
            metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(metadata_path, metadata)
