"""Streaming model gateway for the vLLM servers in ./cluster."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Sequence
import logging
from time import monotonic

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from intelligent_agents_chat.logging_config import log_event, sanitized_endpoint
from intelligent_agents_chat.models import ModelProfile


API_KEY = "not-needed"
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_TOKENS = 1024
THINKING_MAX_TOKENS = 8192
TEMPERATURE = 0.2
SYSTEM_PROMPT = "You are a helpful assistant. Give clear, accurate, and concise answers."

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """A safe, user-facing model generation error."""


class VLLMGateway:
    """Create streamed replies with the selected vLLM profile."""

    async def stream_reply(
        self,
        profile: ModelProfile,
        messages: Sequence[dict[str, str]],
        *,
        request_id: str | None = None,
        thinking_enabled: bool = False,
        tools: Sequence[dict] | None = None,
    ) -> AsyncIterator[str]:
        started_at = monotonic()
        chunk_count = 0
        output_chars = 0
        reasoning_chunk_count = 0
        reasoning_chars = 0
        tool_call_chunk_count = 0
        first_chunk_ms: float | None = None
        first_reasoning_chunk_ms: float | None = None
        server_response_id: str | None = None
        finish_reason: str | None = None
        outcome = "started"
        client: AsyncOpenAI | None = None
        stream = None
        effective_thinking = thinking_enabled and profile.supports_thinking
        max_tokens = THINKING_MAX_TOKENS if effective_thinking else MAX_TOKENS
        context = {
            "request_id": request_id,
            "profile_key": profile.key,
            "model_name": profile.model,
            "endpoint": sanitized_endpoint(profile.base_url),
            "supports_thinking": profile.supports_thinking,
            "thinking_enabled": effective_thinking,
            "max_tokens": max_tokens,
            "input_message_count": len(messages),
            "input_chars": sum(len(message.get("content", "")) for message in messages),
            "input_role_counts": dict(
                Counter(message.get("role", "unknown") for message in messages)
            ),
            "tool_count": len(tools) if tools else 0,
        }
        log_event(logger, logging.INFO, "llm.stream.started", **context)

        try:
            client = AsyncOpenAI(
                base_url=profile.base_url,
                api_key=API_KEY,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            extra_body = (
                {"chat_template_kwargs": {"enable_thinking": effective_thinking}}
                if profile.supports_thinking
                else None
            )
            create_kwargs: dict[str, object] = {
                "model": profile.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": TEMPERATURE,
                "stream": True,
                "extra_body": extra_body,
            }
            if tools:
                create_kwargs["tools"] = tools
            stream = await client.chat.completions.create(**create_kwargs)
            reasoning_open = False
            async for chunk in stream:
                if chunk.id:
                    server_response_id = chunk.id
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = str(choice.finish_reason)
                reasoning = getattr(choice.delta, "reasoning", None) or getattr(
                    choice.delta,
                    "reasoning_content",
                    None,
                )
                if reasoning:
                    if first_reasoning_chunk_ms is None:
                        first_reasoning_chunk_ms = round(
                            (monotonic() - started_at) * 1000,
                            2,
                        )
                    reasoning_chunk_count += 1
                    reasoning_chars += len(reasoning)
                    fragment = reasoning if reasoning_open else f"<think>\n{reasoning}"
                    reasoning_open = True
                    if first_chunk_ms is None:
                        first_chunk_ms = round((monotonic() - started_at) * 1000, 2)
                    chunk_count += 1
                    output_chars += len(fragment)
                    yield fragment
                content = choice.delta.content
                if content:
                    if reasoning_open:
                        content = f"\n</think>\n\n{content}"
                        reasoning_open = False
                    if first_chunk_ms is None:
                        first_chunk_ms = round((monotonic() - started_at) * 1000, 2)
                    chunk_count += 1
                    output_chars += len(content)
                    yield content
                # No tool execution yet -- surface the raw tool-call deltas as
                # visible text instead of silently dropping them.
                for tool_call in getattr(choice.delta, "tool_calls", None) or []:
                    function = getattr(tool_call, "function", None)
                    fragment = ""
                    if reasoning_open:
                        fragment += "\n</think>\n\n"
                        reasoning_open = False
                    if function is not None and function.name:
                        fragment += f"\n[tool_call: {function.name}] "
                    if function is not None and function.arguments:
                        fragment += function.arguments
                    if fragment:
                        if first_chunk_ms is None:
                            first_chunk_ms = round((monotonic() - started_at) * 1000, 2)
                        chunk_count += 1
                        tool_call_chunk_count += 1
                        output_chars += len(fragment)
                        yield fragment
            outcome = "completed"
        except asyncio.CancelledError:
            outcome = "cancelled"
            log_event(logger, logging.WARNING, "llm.stream.cancelled", **context)
            raise
        except GeneratorExit:
            outcome = "closed"
            raise
        except (APIConnectionError, APITimeoutError) as error:
            outcome = "connection_error"
            logger.exception(
                "llm.stream.connection_error",
                extra={"event": "llm.stream.connection_error", **context},
            )
            raise LLMError(
                f"Could not reach the {profile.label} vLLM endpoint. "
                "Check the server or SSH tunnel and try again."
            ) from error
        except APIStatusError as error:
            outcome = "http_error"
            logger.exception(
                "llm.stream.http_error",
                extra={
                    "event": "llm.stream.http_error",
                    **context,
                    "http_status": error.status_code,
                },
            )
            raise LLMError(
                f"The {profile.label} vLLM endpoint returned HTTP {error.status_code}."
            ) from error
        finally:
            try:
                if stream is not None:
                    await stream.close()
            except Exception:
                outcome = "stream_cleanup_error"
                logger.exception(
                    "llm.stream.cleanup_error",
                    extra={
                        "event": "llm.stream.cleanup_error",
                        **context,
                        "resource": "response_stream",
                    },
                )
                raise
            finally:
                try:
                    if client is not None:
                        await client.close()
                except Exception:
                    outcome = "client_cleanup_error"
                    logger.exception(
                        "llm.stream.cleanup_error",
                        extra={
                            "event": "llm.stream.cleanup_error",
                            **context,
                            "resource": "openai_client",
                        },
                    )
                    raise
                finally:
                    log_event(
                        logger,
                        logging.INFO,
                        "llm.stream.finished",
                        **context,
                        outcome=outcome,
                        chunk_count=chunk_count,
                        output_chars=output_chars,
                        reasoning_chunk_count=reasoning_chunk_count,
                        reasoning_chars=reasoning_chars,
                        tool_call_chunk_count=tool_call_chunk_count,
                        time_to_first_chunk_ms=first_chunk_ms,
                        time_to_first_reasoning_chunk_ms=first_reasoning_chunk_ms,
                        server_response_id=server_response_id,
                        finish_reason=finish_reason,
                        duration_ms=round((monotonic() - started_at) * 1000, 2),
                    )
