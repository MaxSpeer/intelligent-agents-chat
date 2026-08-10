"""Streaming model gateway for the built-in test model and vLLM servers."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Sequence
import logging
from time import monotonic

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from intelligent_agents_chat.config import ModelProfile, Settings
from intelligent_agents_chat.logging_config import log_event, sanitized_endpoint


LOREM_IPSUM_RESPONSE = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua."
)
logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """A safe, user-facing model generation error."""


class VLLMGateway:
    """Create streamed replies with the selected local or vLLM profile."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def stream_reply(
        self,
        profile: ModelProfile,
        messages: Sequence[dict[str, str]],
        *,
        request_id: str | None = None,
        thinking_enabled: bool = False,
    ) -> AsyncIterator[str]:
        started_at = monotonic()
        chunk_count = 0
        output_chars = 0
        reasoning_chunk_count = 0
        reasoning_chars = 0
        first_chunk_ms: float | None = None
        first_reasoning_chunk_ms: float | None = None
        server_response_id: str | None = None
        finish_reason: str | None = None
        outcome = "started"
        client: AsyncOpenAI | None = None
        stream = None
        effective_thinking = thinking_enabled and profile.supports_thinking
        max_tokens = (
            self.settings.thinking_max_tokens if effective_thinking else self.settings.max_tokens
        )
        context = {
            "request_id": request_id,
            "profile_key": profile.key,
            "profile_backend": profile.backend,
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
        }
        log_event(logger, logging.INFO, "llm.stream.started", **context)

        try:
            if profile.backend == "lorem":
                for index, word in enumerate(LOREM_IPSUM_RESPONSE.split()):
                    await asyncio.sleep(0.02)
                    chunk = word if index == 0 else f" {word}"
                    if first_chunk_ms is None:
                        first_chunk_ms = round((monotonic() - started_at) * 1000, 2)
                    chunk_count += 1
                    output_chars += len(chunk)
                    yield chunk
                outcome = "completed"
                return

            if profile.backend != "vllm" or not profile.base_url or not profile.model:
                outcome = "invalid_profile"
                raise LLMError(f"The {profile.label} model profile is not configured correctly.")

            client = AsyncOpenAI(
                base_url=profile.base_url,
                api_key=self.settings.api_key,
                timeout=self.settings.request_timeout_seconds,
            )
            extra_body = (
                {"chat_template_kwargs": {"enable_thinking": effective_thinking}}
                if profile.supports_thinking
                else None
            )
            stream = await client.chat.completions.create(
                model=profile.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.settings.temperature,
                stream=True,
                extra_body=extra_body,
            )
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
                content = choice.delta.content
                if content:
                    if first_chunk_ms is None:
                        first_chunk_ms = round((monotonic() - started_at) * 1000, 2)
                    chunk_count += 1
                    output_chars += len(content)
                    yield content
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
                        time_to_first_chunk_ms=first_chunk_ms,
                        time_to_first_reasoning_chunk_ms=first_reasoning_chunk_ms,
                        server_response_id=server_response_id,
                        finish_reason=finish_reason,
                        duration_ms=round((monotonic() - started_at) * 1000, 2),
                    )
