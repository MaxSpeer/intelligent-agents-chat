"""Integration-style test for the OpenAI-compatible streaming gateway."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
import time
import unittest

from intelligent_agents_chat.llm import (
    MAX_TOKENS,
    THINKING_MAX_TOKENS,
    VLLMGateway,
    check_model_available,
)
from intelligent_agents_chat.models import ModelProfile


async def _collect_text(stream) -> str:
    return "".join([delta.text async for delta in stream])


class FakeVLLMHandler(BaseHTTPRequestHandler):
    request_body: dict | None = None

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/models":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        body = json.dumps({"data": [{"id": "test-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        FakeVLLMHandler.request_body = json.loads(self.rfile.read(content_length))
        if not FakeVLLMHandler.request_body.get("stream", False):
            body = json.dumps(
                {
                    "id": "chatcmpl-control",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": '{"retrieve":false,"query":"","reason":"skip"}',
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
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

        texts = [
            delta.text
            async for delta in VLLMGateway().stream_reply(
                profile,
                [{"role": "user", "content": "Hello"}],
            )
        ]

        self.assertEqual(texts, ["Hello ", "world"])
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

    async def test_control_completion_is_non_streaming_and_disables_thinking(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="control",
            label="Control model",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
            supports_thinking=True,
        )

        response = await VLLMGateway().complete_control(
            profile,
            [{"role": "user", "content": "Route this"}],
            purpose="test_router",
            request_id="request-1",
            max_tokens=123,
        )

        self.assertIn('"retrieve":false', response)
        self.assertIsNotNone(FakeVLLMHandler.request_body)
        assert FakeVLLMHandler.request_body is not None
        self.assertFalse(FakeVLLMHandler.request_body["stream"])
        self.assertEqual(FakeVLLMHandler.request_body["max_tokens"], 123)
        self.assertEqual(FakeVLLMHandler.request_body["temperature"], 0.0)
        self.assertEqual(
            FakeVLLMHandler.request_body["chat_template_kwargs"],
            {"enable_thinking": False},
        )

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

    async def test_tool_call_deltas_are_reconstructed_not_shown_as_text(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="test",
            label="Test model",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )
        tool_calls: list[dict] = []

        response = await _collect_text(
            VLLMGateway().stream_reply(
                profile,
                [{"role": "user", "content": "Tool call check"}],
                tool_calls=tool_calls,
            )
        )

        # No decorative text for tool calls -- only the structured reconstruction.
        self.assertEqual(response, "")
        self.assertEqual(
            tool_calls,
            [{"id": "call_1", "name": "calculator", "arguments": '{"expression": "2+2"}'}],
        )

    async def test_reasoning_deltas_are_tagged_as_reasoning(self) -> None:
        # No decorative wrapping here (no "**Thinking:**" header, no "---"
        # separator) -- chat.py's format_reasoning_entry() adds that framing
        # once, for the accordion entry; llm.py just tags raw text.
        host, port = self.server.server_address
        profile = ModelProfile(
            key="test",
            label="Test model",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )

        deltas = [
            delta
            async for delta in VLLMGateway().stream_reply(
                profile,
                [{"role": "user", "content": "Reasoning check"}],
            )
        ]

        self.assertEqual("".join(delta.text for delta in deltas), "Let me think.42")
        self.assertEqual([delta.is_reasoning for delta in deltas], [True, True, False])

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

        self.assertEqual((await anext(stream)).text, "Hello ")
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

        response = await _collect_text(
            VLLMGateway().stream_reply(
                profile,
                [{"role": "user", "content": "Think"}],
                thinking_enabled=True,
            )
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

    async def test_profile_can_set_reasoning_effort_for_ollama(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="ollama",
            label="Local Ollama",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
            reasoning_effort="none",
        )

        _ = [
            chunk
            async for chunk in VLLMGateway().stream_reply(
                profile,
                [{"role": "user", "content": "Answer directly"}],
            )
        ]

        assert FakeVLLMHandler.request_body is not None
        self.assertEqual(FakeVLLMHandler.request_body["reasoning_effort"], "none")
        self.assertNotIn("chat_template_kwargs", FakeVLLMHandler.request_body)


class CheckModelAvailableTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLMHandler)
        cls.server_thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=2)

    async def test_returns_true_when_the_endpoint_serves_the_model(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="test",
            label="Test model",
            base_url=f"http://{host}:{port}/v1",
            model="test-model",
        )

        self.assertTrue(await check_model_available(profile))

    async def test_returns_false_when_the_endpoint_does_not_serve_the_model(self) -> None:
        host, port = self.server.server_address
        profile = ModelProfile(
            key="other",
            label="Other model",
            base_url=f"http://{host}:{port}/v1",
            model="some-other-model",
        )

        self.assertFalse(await check_model_available(profile))

    async def test_returns_false_when_the_endpoint_is_unreachable(self) -> None:
        profile = ModelProfile(
            key="down",
            label="Down model",
            base_url="http://127.0.0.1:1/v1",
            model="test-model",
        )

        self.assertFalse(await check_model_available(profile))


if __name__ == "__main__":
    unittest.main()
