"""OpenAI-compatible embedding configuration and transport tests."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest
from unittest.mock import patch

from intelligent_agents_chat.embeddings import (
    EmbeddingSettings,
    OpenAIEmbeddingGateway,
    create_embedding_gateway,
)


class FakeEmbeddingHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(content_length))
        inputs = request["input"]
        payload = {
            "object": "list",
            "model": request["model"],
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": [float(len(text)), float(index + 1)],
                }
                for index, text in enumerate(inputs)
            ],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args) -> None:  # noqa: A002
        return


class EmbeddingGatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEmbeddingHandler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    async def test_openai_compatible_gateway_batches_and_preserves_order(self) -> None:
        gateway = OpenAIEmbeddingGateway(
            EmbeddingSettings(
                base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
                model="embed-test",
                api_key="placeholder",
                batch_size=1,
                expected_dimension=2,
            )
        )

        vectors = await gateway.embed(["alpha", "longer"])

        self.assertEqual(vectors, [[5.0, 1.0], [6.0, 1.0]])


class EmbeddingSettingsTests(unittest.TestCase):
    def test_endpoint_is_independent_and_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = EmbeddingSettings.from_environment()
            gateway = create_embedding_gateway(settings)

        self.assertFalse(settings.enabled)
        self.assertIsNone(gateway)

    def test_endpoint_and_model_must_be_configured_together(self) -> None:
        with patch.dict(
            os.environ,
            {"RAG_EMBEDDING_BASE_URL": "http://127.0.0.1:11434/v1"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "configured together"):
                EmbeddingSettings.from_environment()


if __name__ == "__main__":
    unittest.main()
