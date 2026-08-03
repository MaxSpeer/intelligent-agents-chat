"""Streaming model gateway for the built-in test model and vLLM servers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from intelligent_agents_chat.config import ModelProfile, Settings


LOREM_IPSUM_RESPONSE = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua."
)


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
    ) -> AsyncIterator[str]:
        if profile.backend == "lorem":
            for index, word in enumerate(LOREM_IPSUM_RESPONSE.split()):
                await asyncio.sleep(0.02)
                yield word if index == 0 else f" {word}"
            return

        if profile.backend != "vllm" or not profile.base_url or not profile.model:
            raise LLMError(f"The {profile.label} model profile is not configured correctly.")

        client = AsyncOpenAI(
            base_url=profile.base_url,
            api_key=self.settings.api_key,
            timeout=self.settings.request_timeout_seconds,
        )
        stream = None
        try:
            stream = await client.chat.completions.create(
                model=profile.model,
                messages=messages,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except (APIConnectionError, APITimeoutError) as error:
            raise LLMError(
                f"Could not reach the {profile.label} vLLM endpoint. "
                "Check the server or SSH tunnel and try again."
            ) from error
        except APIStatusError as error:
            raise LLMError(
                f"The {profile.label} vLLM endpoint returned HTTP {error.status_code}."
            ) from error
        finally:
            if stream is not None:
                await stream.close()
            await client.close()
