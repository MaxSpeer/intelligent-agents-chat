"""Local (fastembed-backed) embedding gateway tests.

Injects a fake stand-in for fastembed's TextEmbedding (by setting the
gateway's private _model directly, bypassing _ensure_model's real load path)
so these tests never do real ONNX inference or a model download.
"""

import unittest

import numpy as np

from intelligent_agents_chat.embeddings import (
    EMBEDDING_MODEL_NAME,
    EmbeddingError,
    LocalEmbeddingGateway,
    create_embedding_gateway,
)


class _FakeFastEmbedModel:
    """Stands in for fastembed's TextEmbedding -- one small vector per input
    text, no real ONNX inference or model download.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def embed(self, documents):
        if self.fail:
            raise RuntimeError("boom")
        texts = list(documents)
        self.calls.append(texts)
        return [np.array([float(len(text)), 1.0, 0.0]) for text in texts]


class LocalEmbeddingGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_embeds_texts_via_the_model_preserving_order(self) -> None:
        gateway = LocalEmbeddingGateway()
        fake_model = _FakeFastEmbedModel()
        gateway._model = fake_model  # bypass the real fastembed load for this test

        vectors = await gateway.embed(["a", "bb"])

        self.assertEqual(vectors, [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
        self.assertEqual(fake_model.calls, [["a", "bb"]])

    async def test_empty_input_short_circuits_without_touching_the_model(self) -> None:
        gateway = LocalEmbeddingGateway()
        vectors = await gateway.embed([])
        self.assertEqual(vectors, [])

    async def test_model_failure_is_reported_as_embedding_error(self) -> None:
        gateway = LocalEmbeddingGateway()
        gateway._model = _FakeFastEmbedModel(fail=True)

        with self.assertRaisesRegex(EmbeddingError, "embedding failed"):
            await gateway.embed(["x"])

    def test_model_name_defaults_to_the_pinned_lightweight_model(self) -> None:
        gateway = LocalEmbeddingGateway()
        self.assertEqual(gateway.model_name, EMBEDDING_MODEL_NAME)


class CreateEmbeddingGatewayTests(unittest.TestCase):
    def test_always_returns_a_real_local_gateway(self) -> None:
        # Unlike the old remote-endpoint gateway, there's no "unconfigured"
        # state any more (no env vars to set) -- this never returns None.
        gateway = create_embedding_gateway()
        self.assertIsInstance(gateway, LocalEmbeddingGateway)
        self.assertEqual(gateway.model_name, EMBEDDING_MODEL_NAME)


if __name__ == "__main__":
    unittest.main()
