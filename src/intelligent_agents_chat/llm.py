"""Streaming gateway for OpenAI-compatible model servers."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import date
import logging
from time import monotonic

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from intelligent_agents_chat.logging_config import log_event, sanitized_endpoint
from intelligent_agents_chat.models import ModelProfile


API_KEY = "not-needed"
REQUEST_TIMEOUT_SECONDS = 120.0
CONTROL_REQUEST_TIMEOUT_SECONDS = 60.0
HEALTH_CHECK_TIMEOUT_SECONDS = 3.0
MAX_TOKENS = 1024
THINKING_MAX_TOKENS = 8192
TEMPERATURE = 0.2
SYSTEM_PROMPT = (
    "You are a helpful assistant. Give clear, accurate, and concise answers. "
    "When you use tools, don't settle for a thin or inconclusive first result -- if a "
    "web search's snippets don't clearly answer the question, fetch the most promising "
    "page for more detail before giving your final answer."
)


def system_prompt_for_today() -> str:
    """SYSTEM_PROMPT with today's real date spliced in, computed fresh on
    every call (never cached) so each request tells the model what day it
    actually is. Without this, a model doesn't reach for a tool to check --
    it isn't *unsure* what day it is, it's *confidently wrong* (its training
    cutoff), so nothing prompts it to ever question that. The date has to be
    stated unconditionally, not offered as something optional to look up.
    """
    return f"Today's date is {date.today().isoformat()}. {SYSTEM_PROMPT}"

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """A safe, user-facing model generation error."""


async def check_model_available(profile: ModelProfile) -> bool:
    """Return whether the profile's vLLM endpoint is reachable and serving its model."""
    client = AsyncOpenAI(
        base_url=profile.base_url,
        api_key=API_KEY,
        timeout=HEALTH_CHECK_TIMEOUT_SECONDS,
    )
    try:
        response = await client.models.list()
    except Exception:
        return False
    finally:
        await client.close()
    return any(model.id == profile.model for model in response.data)



@dataclass(frozen=True, slots=True)
class ContentDelta:
    """One fragment of visible model text, tagged so callers can tell reasoning
    (show it, but never feed it back as conversation history) from the model's
    actual answer content (show it, and it belongs in history).
    """
    text: str
    is_reasoning: bool = False


class VLLMGateway:
    """Create streamed replies with the selected OpenAI-compatible profile."""

    async def complete_control(
        self,
        profile: ModelProfile,
        messages: Sequence[dict[str, str]],
        *,
        purpose: str,
        request_id: str | None = None,
        max_tokens: int = 384,
    ) -> str:
        """Run a short non-streaming control request with model reasoning disabled."""
        if max_tokens <= 0:
            raise ValueError("Control max_tokens must be positive")
        started_at = monotonic()
        client: AsyncOpenAI | None = None
        context = {
            "request_id": request_id,
            "purpose": purpose,
            "profile_key": profile.key,
            "model_name": profile.model,
            "endpoint": sanitized_endpoint(profile.base_url),
            "max_tokens": max_tokens,
            "input_message_count": len(messages),
            "input_chars": sum(len(message.get("content", "")) for message in messages),
        }
        log_event(logger, logging.INFO, "llm.control.started", **context)
        outcome = "started"
        output = ""
        finish_reason: str | None = None
        try:
            client = AsyncOpenAI(
                base_url=profile.base_url,
                api_key=API_KEY,
                timeout=CONTROL_REQUEST_TIMEOUT_SECONDS,
            )
            extra_body: dict[str, object] = {}
            if profile.supports_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            if profile.reasoning_effort is not None:
                extra_body["reasoning_effort"] = profile.reasoning_effort
            response = await client.chat.completions.create(
                model=profile.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                stream=False,
                extra_body=extra_body or None,
            )
            if not response.choices:
                raise LLMError(f"{profile.label} returned no control response.")
            choice = response.choices[0]
            finish_reason = str(choice.finish_reason) if choice.finish_reason is not None else None
            output = choice.message.content or ""
            if not output.strip():
                raise LLMError(f"{profile.label} returned an empty control response.")
            outcome = "completed"
            return output
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except (APIConnectionError, APITimeoutError) as error:
            outcome = "connection_error"
            raise LLMError(
                f"Could not reach the {profile.label} model endpoint for {purpose}."
            ) from error
        except APIStatusError as error:
            outcome = "http_error"
            raise LLMError(
                f"The {profile.label} model endpoint returned HTTP {error.status_code} for "
                f"{purpose}."
            ) from error
        except Exception:
            outcome = "control_error"
            raise
        finally:
            if client is not None:
                await client.close()
            log_event(
                logger,
                logging.INFO,
                "llm.control.finished",
                **context,
                outcome=outcome,
                output_chars=len(output),
                finish_reason=finish_reason,
                duration_ms=round((monotonic() - started_at) * 1_000, 2),
            )

    async def stream_reply(
        self,
        profile: ModelProfile,
        messages: Sequence[dict[str, object]],
        *,
        request_id: str | None = None,
        thinking_enabled: bool = False,
        tools: Sequence[dict] | None = None,
        tool_calls: list[dict] | None = None,
    ) -> AsyncIterator[ContentDelta]:
        """Stream the reply as tagged text fragments; if `tool_calls` is given, append
        the fully reconstructed tool calls to it (id, name, arguments) once the stream
        completes.
        """
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
            "reasoning_effort": profile.reasoning_effort,
            "max_tokens": max_tokens,
            "input_message_count": len(messages),
            "input_chars": sum(len(message.get("content") or "") for message in messages),
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
            extra_body: dict[str, object] = {}
            if profile.supports_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": effective_thinking}
            if profile.reasoning_effort is not None:
                extra_body["reasoning_effort"] = profile.reasoning_effort
            create_kwargs: dict[str, object] = {
                "model": profile.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": TEMPERATURE,
                "stream": True,
                "extra_body": extra_body or None,
            }
            if tools:
                create_kwargs["tools"] = tools
            stream = await client.chat.completions.create(**create_kwargs)
            reasoning_open = False
            tool_call_buffers: dict[int, dict[str, str | None]] = {}
            async for chunk in stream:
                if chunk.id:
                    server_response_id = chunk.id
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = str(choice.finish_reason)
                reasoning = getattr(choice.delta, "reasoning", None) or getattr(choice.delta, "reasoning_content", None)
                if reasoning:
                    if first_reasoning_chunk_ms is None:
                        first_reasoning_chunk_ms = round(
                            (monotonic() - started_at) * 1000,
                            2,
                        )
                    reasoning_chunk_count += 1
                    reasoning_chars += len(reasoning)
                    reasoning_open = True
                    if first_chunk_ms is None:
                        first_chunk_ms = round((monotonic() - started_at) * 1000, 2)
                    chunk_count += 1
                    output_chars += len(reasoning)
                    yield ContentDelta(reasoning, is_reasoning=True)
                content = choice.delta.content
                if content:
                    if reasoning_open:
                        reasoning_open = False
                    if first_chunk_ms is None:
                        first_chunk_ms = round((monotonic() - started_at) * 1000, 2)
                    chunk_count += 1
                    output_chars += len(content)
                    yield ContentDelta(content)
                # Reconstruct tool-call deltas (id, name, arguments) for the caller to
                # execute -- no decorative text here; chat.py renders tool calls/results
                # from this structured data instead, since it persists them structurally.
                for tool_call_delta in getattr(choice.delta, "tool_calls", None) or []:
                    buffer = tool_call_buffers.setdefault(
                        tool_call_delta.index, {"id": None, "name": None, "arguments": ""}
                    )
                    if tool_call_delta.id:
                        buffer["id"] = tool_call_delta.id
                    function = getattr(tool_call_delta, "function", None)
                    if function is not None and function.name:
                        buffer["name"] = function.name
                    if function is not None and function.arguments:
                        buffer["arguments"] = (buffer["arguments"] or "") + function.arguments
                    tool_call_chunk_count += 1
            outcome = "completed"
            if tool_calls is not None:
                tool_calls.extend(tool_call_buffers.values())
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
                f"Could not reach the {profile.label} model endpoint. "
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
                f"The {profile.label} model endpoint returned HTTP {error.status_code}."
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
