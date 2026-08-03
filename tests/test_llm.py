"""Integration-style test for the OpenAI-compatible streaming gateway."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
import time
import unittest

from intelligent_agents_chat.config import ModelProfile, Settings
from intelligent_agents_chat.llm import LOREM_IPSUM_RESPONSE, VLLMGateway


class FakeVLLMHandler(BaseHTTPRequestHandler):
    request_body: dict | None = None

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        FakeVLLMHandler.request_body = json.loads(self.rfile.read(content_length))
        messages = FakeVLLMHandler.request_body.get("messages", [])
        slow_response = any(message.get("content") == "Slow stream check" for message in messages)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for index, (content, finish_reason) in enumerate((("Hello ", None), ("world", "stop"))):
            payload = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
            if slow_response and index == 0:
                time.sleep(1)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        """Keep the test output quiet."""


class VLLMGatewayTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FakeVLLMHandler.request_body = None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLMHandler)
        cls.server_thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)

    async def test_stream_reply_uses_the_openai_compatible_contract(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="test",
            label="Test model",
            backend="vllm",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )
        settings = Settings(
            database_path=Path("unused.sqlite3"),
            model_profiles=(profile,),
            default_profile_key=profile.key,
            api_key="not-needed",
            request_timeout_seconds=2,
            max_tokens=64,
            temperature=0.1,
            system_prompt="Test system prompt",
        )

        chunks = [
            chunk
            async for chunk in VLLMGateway(settings).stream_reply(
                profile,
                [{"role": "user", "content": "Hello"}],
            )
        ]

        self.assertEqual(chunks, ["Hello ", "world"])
        self.assertIsNotNone(FakeVLLMHandler.request_body)
        assert FakeVLLMHandler.request_body is not None
        self.assertEqual(FakeVLLMHandler.request_body["model"], "test-model")
        self.assertEqual(
            FakeVLLMHandler.request_body["messages"],
            [{"role": "user", "content": "Hello"}],
        )
        self.assertTrue(FakeVLLMHandler.request_body["stream"])

        await asyncio.sleep(0)

    async def test_stream_reply_can_be_cancelled_while_waiting_for_a_chunk(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="slow",
            label="Slow test model",
            backend="vllm",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )
        settings = Settings(
            database_path=Path("unused.sqlite3"),
            model_profiles=(profile,),
            default_profile_key=profile.key,
            api_key="not-needed",
            request_timeout_seconds=2,
            max_tokens=64,
            temperature=0.1,
            system_prompt="Test system prompt",
        )
        stream = VLLMGateway(settings).stream_reply(
            profile,
            [{"role": "user", "content": "Slow stream check"}],
        )

        self.assertEqual(await anext(stream), "Hello ")
        next_chunk = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.05)
        next_chunk.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await next_chunk
        await stream.aclose()

    async def test_lorem_profile_streams_a_deterministic_reply_without_a_server(self) -> None:
        settings = Settings.from_env({})
        profile = settings.profile("lorem")

        response = "".join(
            [
                chunk
                async for chunk in VLLMGateway(settings).stream_reply(
                    profile,
                    [{"role": "user", "content": "Any prompt produces the same reply"}],
                )
            ]
        )

        self.assertEqual(response, LOREM_IPSUM_RESPONSE)


if __name__ == "__main__":
    unittest.main()
