`qwen3-chat-template.jinja` is the unmodified Qwen3 template from the tokenizer
snapshot recorded in `training/datasets/plain_english/manifest.json`.

Upstream: [Qwen/Qwen3-8B, pinned revision](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218).
License: Apache-2.0, as declared by that model repository. The template SHA-256 is
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.

This fixture reproduces the historical/final assistant formatting difference
without model weights. Set `QWEN_TOKENIZER_PATH` to the pinned local tokenizer
snapshot to also run tests with real token IDs across all 118 dataset answers.
