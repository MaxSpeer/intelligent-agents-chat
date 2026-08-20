"""Integration-style test for the OpenAI-compatible streaming gateway."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
import time
import unittest

from intelligent_agents_chat.llm import MAX_TOKENS, THINKING_MAX_TOKENS, VLLMGateway
from intelligent_agents_chat.models import ModelProfile


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
        tool_call_response = any(
            message.get("content") == "Tool call check" for message in messages
        )
        reasoning_response = any(
            message.get("content") == "Reasoning check" for message in messages
        )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if reasoning_response:
            deltas: list[tuple[dict, str | None]] = [
                ({"reasoning_content": "Let me "}, None),
                ({"reasoning_content": "think."}, None),
                ({"content": "42"}, "stop"),
            ]
        elif tool_call_response:
            deltas = [
                (
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "calculator", "arguments": ""},
                            }
                        ]
                    },
                    None,
                ),
                (
                    {"tool_calls": [{"index": 0, "function": {"arguments": '{"expression"'}}]},
                    None,
                ),
                (
                    {"tool_calls": [{"index": 0, "function": {"arguments": ': "2+2"}'}}]},
                    "tool_calls",
                ),
            ]
        else:
            deltas = [
                ({"content": "Hello "}, None),
                ({"content": "world"}, "stop"),
            ]
        for index, (delta, finish_reason) in enumerate(deltas):
            payload = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
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
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )

        chunks = [
            chunk
            async for chunk in VLLMGateway().stream_reply(
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
        self.assertEqual(FakeVLLMHandler.request_body["max_tokens"], MAX_TOKENS)
        self.assertNotIn("chat_template_kwargs", FakeVLLMHandler.request_body)
        self.assertNotIn("tools", FakeVLLMHandler.request_body)

        await asyncio.sleep(0)

    async def test_tools_are_forwarded_when_provided(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="test",
            label="Test model",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )
        calculator_tool = {
            "type": "function",
            "function": {"name": "calculator", "description": "Evaluate an expression."},
        }

        _ = [
            chunk
            async for chunk in VLLMGateway().stream_reply(
                profile,
                [{"role": "user", "content": "What is 2 + 2?"}],
                tools=[calculator_tool],
            )
        ]

        assert FakeVLLMHandler.request_body is not None
        self.assertEqual(FakeVLLMHandler.request_body["tools"], [calculator_tool])

    async def test_tool_call_deltas_are_surfaced_as_output_text(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="test",
            label="Test model",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )

        response = "".join(
            [
                chunk
                async for chunk in VLLMGateway().stream_reply(
                    profile,
                    [{"role": "user", "content": "Tool call check"}],
                )
            ]
        )

        self.assertEqual(response, '\n[tool_call: calculator] {"expression": "2+2"}')

    async def test_reasoning_deltas_are_wrapped_and_surfaced_as_output_text(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="test",
            label="Test model",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )

        response = "".join(
            [
                chunk
                async for chunk in VLLMGateway().stream_reply(
                    profile,
                    [{"role": "user", "content": "Reasoning check"}],
                )
            ]
        )

        self.assertEqual(response, "<think>\nLet me think.\n</think>\n\n42")

    async def test_stream_reply_can_be_cancelled_while_waiting_for_a_chunk(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="slow",
            label="Slow test model",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )
        stream = VLLMGateway().stream_reply(
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

    async def test_thinking_profile_uses_profile_specific_flag_and_token_limit(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="thinking",
            label="Thinking model",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
            supports_thinking=True,
        )

        response = "".join(
            [
                chunk
                async for chunk in VLLMGateway().stream_reply(
                    profile,
                    [{"role": "user", "content": "Think"}],
                    thinking_enabled=True,
                )
            ]
        )

        self.assertEqual(response, "Hello world")
        self.assertIsNotNone(FakeVLLMHandler.request_body)
        assert FakeVLLMHandler.request_body is not None
        self.assertEqual(FakeVLLMHandler.request_body["max_tokens"], THINKING_MAX_TOKENS)
        self.assertEqual(
            FakeVLLMHandler.request_body["chat_template_kwargs"],
            {"enable_thinking": True},
        )

        _ = [
            chunk
            async for chunk in VLLMGateway().stream_reply(
                profile,
                [{"role": "user", "content": "Do not think"}],
                thinking_enabled=False,
            )
        ]
        assert FakeVLLMHandler.request_body is not None
        self.assertEqual(FakeVLLMHandler.request_body["max_tokens"], MAX_TOKENS)
        self.assertEqual(
            FakeVLLMHandler.request_body["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    async def test_profile_without_thinking_support_ignores_the_thinking_flag(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="no-thinking",
            label="Non-thinking model",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )

        _ = [
            chunk
            async for chunk in VLLMGateway().stream_reply(
                profile,
                [{"role": "user", "content": "Think anyway"}],
                thinking_enabled=True,
            )
        ]

        assert FakeVLLMHandler.request_body is not None
        self.assertEqual(FakeVLLMHandler.request_body["max_tokens"], MAX_TOKENS)
        self.assertNotIn("chat_template_kwargs", FakeVLLMHandler.request_body)


if __name__ == "__main__":
    unittest.main()
