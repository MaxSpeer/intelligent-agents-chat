"""No GPU or trained model needed: exercise history isolation and callback state."""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
AVAILABLE = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft"))
if AVAILABLE:
    import torch
    from validation_samples import ValidationSamplesCallback


@unittest.skipUnless(AVAILABLE, "Requires the training Python environment, CPU only")
class ValidationSamplesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = [{"id": "validation-1", "messages": [
            {"role": "system", "content": "Neutral system."},
            {"role": "user", "content": "First question?"},
            {"role": "assistant", "content": "GOLD REFERENCE ONE"},
            {"role": "user", "content": "Follow-up question?"},
            {"role": "assistant", "content": "GOLD REFERENCE TWO"},
        ]}]
        self.state = SimpleNamespace(global_step=6, epoch=1.0, log_history=[{"eval_loss": 2.0}])
        histories = []
        class Encoded(dict):
            def to(self, device):
                return self
        class Tokenizer:
            pad_token_id = 0
            def apply_chat_template(self, history, **kwargs):
                histories.append(copy.deepcopy(history))
                assert kwargs["enable_thinking"] is False
                return Encoded(input_ids=torch.tensor([[1, 2]]))
            def decode(self, ids, **kwargs):
                return "".join(chr(i) for i in ids if i)
        class Model:
            training = True
            adapter_enabled = True
            device = torch.device("cpu")
            generation_config = SimpleNamespace(eos_token_id=0)
            fail = False
            def train(self, mode=True):
                self.training = mode
                return self
            def eval(self):
                return self.train(False)
            @contextmanager
            def disable_adapter(self):
                previous = self.adapter_enabled
                self.adapter_enabled = False
                try:
                    yield
                finally:
                    self.adapter_enabled = previous
            def generate(self, input_ids, **kwargs):
                assert not self.training
                torch.rand(3)  # Simulate sampling consuming the generator state.
                if self.fail:
                    raise RuntimeError("generation failed")
                ids = [ord(c) for c in "Own generated answer."] + [0]
                return torch.cat([input_ids, torch.tensor([ids])], dim=1)
        self.histories = histories
        self.model = Model()
        self.callback = ValidationSamplesCallback(self.model, Tokenizer(), self.rows, self.root)

    def test_own_history_no_reference_answers_and_rng_restored(self):
        before = torch.random.get_rng_state().clone()
        self.callback.capture("checkpoint-6", self.state)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(self.model.training)
        self.assertEqual(self.histories[1][-2]["content"], "Own generated answer.")
        self.assertNotIn("GOLD REFERENCE", json.dumps(self.histories))
        metadata = json.loads((self.root / "validation_samples/checkpoint-6.metadata.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["answers"], 2)
        self.assertEqual(metadata["truncated_answers"], 0)

    def test_failure_restores_rng_train_mode_and_adapter(self):
        self.model.fail = True
        before = torch.random.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.callback.capture("base-neutral", self.state, disable_adapter=True)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.adapter_enabled)
        path = self.root / "validation_samples/base-neutral.metadata.json"
        self.assertEqual(json.loads(path.read_text())["status"], "failed")

    def test_two_baselines_and_each_saved_checkpoint_are_recorded(self):
        self.callback.on_train_begin(SimpleNamespace(world_size=1), self.state, None)
        self.callback.on_save(None, self.state, None)
        self.assertEqual(len(list((self.root / "validation_samples").glob("*.jsonl"))), 3)
        self.assertEqual(self.histories[0][0]["content"], "Neutral system.")
        self.assertIn("plain English", self.histories[2][0]["content"])
        self.assertEqual(self.rows[0]["messages"][0]["content"], "Neutral system.")
        with self.assertRaises(FileExistsError):
            self.callback.on_save(None, self.state, None)

    def test_refuses_multiple_processes(self):
        with self.assertRaisesRegex(ValueError, "one training process"):
            self.callback.on_train_begin(SimpleNamespace(world_size=2), self.state, None)


@unittest.skipUnless(AVAILABLE and os.environ.get("QWEN_TOKENIZER_PATH"),
                     "Requires training environment and QWEN_TOKENIZER_PATH")
class CpuIntegrationTests(unittest.TestCase):
    def test_real_dataset_and_padded_labels(self):
        from transformers import AutoTokenizer
        from train_lora import JsonlChatDataset, collate
        tok = AutoTokenizer.from_pretrained(os.environ["QWEN_TOKENIZER_PATH"], local_files_only=True)
        dataset = JsonlChatDataset(TRAINING / "datasets/plain_english/train.jsonl", tok, 1024)
        self.assertEqual(dataset.conversations, 80)
        self.assertEqual(len(dataset), 93)
        items = [dataset[0], dataset[1]]
        batch = collate(items, tok.pad_token_id)
        for i, example in enumerate(items):
            self.assertTrue((batch["labels"][i, :example.prompt_length] == -100).all())
            self.assertTrue((batch["labels"][i, len(example.labels):] == -100).all())
            self.assertEqual(batch["attention_mask"][i].sum().item(), len(example.input_ids))

    def test_installed_qwen_peft_generation_api_on_tiny_random_cpu_model(self):
        # No downloaded model weights, optimizer, training, CUDA, or Slurm job.
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from validation_samples import GENERATION_SETTINGS
        model = get_peft_model(Qwen3ForCausalLM(Qwen3Config(
            vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
            max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                      task_type="CAUSAL_LM"))
        model.eval()
        inputs = torch.tensor([[1, 7, 9]])
        settings = {**GENERATION_SETTINGS, "max_new_tokens": 2, "pad_token_id": 0}
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, **settings)
            with model.disable_adapter():
                baseline = model.generate(input_ids=inputs, **settings)
        self.assertGreater(output.shape[1], inputs.shape[1])
        self.assertGreater(baseline.shape[1], inputs.shape[1])


if __name__ == "__main__":
    unittest.main()
