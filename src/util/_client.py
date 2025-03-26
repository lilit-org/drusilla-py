from __future__ import annotations

import os
from typing import Any

import httpx

from ._types import (
    AsyncDeepSeek,
    AsyncStream,
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ResponseFormat,
)


class DeepSeekClient(AsyncDeepSeek):
    """Implementation of AsyncDeepSeek client using httpx."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or os.getenv("BASE_URL", "http://localhost:11434")).rstrip("/")
        self.organization = organization
        self.project = project
        self.http_client = http_client or httpx.AsyncClient()
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, client: DeepSeekClient) -> None:
            self._client = client
            self.completions = self.Completions(self)

        class Completions:
            def __init__(self, chat: DeepSeekClient.Chat) -> None:
                self._chat = chat

            async def create(
                self,
                model: str,
                messages: list[ChatCompletionMessageParam],
                tools: list[ChatCompletionToolParam] | None = None,
                temperature: float | None = None,
                top_p: float | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                max_tokens: int | None = None,
                tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
                response_format: ResponseFormat | None = None,
                parallel_tool_calls: bool | None = None,
                stream: bool = False,
                stream_options: dict[str, bool] | None = None,
                extra_headers: dict[str, str] | None = None,
            ) -> ChatCompletion | AsyncStream:
                """Create a chat completion with the given parameters."""
                client = self._chat._client
                headers = {
                    "Content-Type": "application/json",
                    **(extra_headers or {}),
                }
                if client.api_key:
                    headers["Authorization"] = f"Bearer {client.api_key}"

                data: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                }

                if tools is not None:
                    data["tools"] = tools
                if temperature is not None:
                    data["temperature"] = temperature
                if top_p is not None:
                    data["top_p"] = top_p
                if frequency_penalty is not None:
                    data["frequency_penalty"] = frequency_penalty
                if presence_penalty is not None:
                    data["presence_penalty"] = presence_penalty
                if max_tokens is not None:
                    data["max_tokens"] = max_tokens
                if tool_choice is not None:
                    data["tool_choice"] = tool_choice
                if response_format is not None:
                    data["response_format"] = response_format
                if parallel_tool_calls is not None:
                    data["parallel_tool_calls"] = parallel_tool_calls
                if stream_options is not None:
                    data["stream_options"] = stream_options

                response = await client.http_client.post(
                    f"{client.base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                )
                response.raise_for_status()

                if stream:
                    return AsyncStream(response.aiter_lines())
                else:
                    return ChatCompletion(**response.json())
