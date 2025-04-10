"""
Client Implementation

This module provides a Python client implementation for interacting with the
DeepSeek API. It offers an asynchronous interface using httpx for making HTTP
requests to the DeepSeek service.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..models.shared import set_default_model_client
from ..util.constants import HEADERS, config, err
from ..util.exceptions import NetworkError
from ..util.types import (
    AsyncDeepSeek,
    AsyncStream,
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionSwordChoiceOptionParam,
    ChatCompletionSwordParam,
    ResponseFormat,
)

########################################################
#           DeepSeek Client
########################################################


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
        self.api_key = api_key or "API_KEY"
        self.base_url = base_url or config.BASE_URL
        self.organization = organization
        self.project = project
        self.http_client = http_client or self._create_http_client()
        self.chat = self.Chat(self)

    def _create_http_client(self) -> httpx.AsyncClient:
        """Create and configure an HTTP client with optimal settings."""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.HTTP_TIMEOUT_TOTAL,
                connect=config.HTTP_TIMEOUT_CONNECT,
                read=config.HTTP_TIMEOUT_READ,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=config.HTTP_MAX_KEEPALIVE_CONNECTIONS,
                max_connections=config.HTTP_MAX_CONNECTIONS,
            ),
        )

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
                temperature: float | None = None,
                top_p: float | None = None,
                max_tokens: int | None = None,
                stream: bool = False,
                extra_headers: dict[str, str] | None = None,
                swords: list[ChatCompletionSwordParam] | None = None,
                sword_choice: ChatCompletionSwordChoiceOptionParam | None = None,
                response_format: ResponseFormat | None = None,
                parallel_sword_calls: bool | None = None,
                stream_options: dict[str, bool] | None = None,
            ) -> ChatCompletion | AsyncStream:
                """Create a chat completion with the given parameters."""
                client = self._chat._client
                headers = {
                    **HEADERS,
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

                optional_params = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "swords": swords,
                    "sword_choice": sword_choice,
                    "response_format": response_format,
                    "parallel_sword_calls": parallel_sword_calls,
                    "stream_options": stream_options,
                }
                data.update({k: v for k, v in optional_params.items() if v is not None})

                endpoint = os.getenv("CHAT_COMPLETIONS_ENDPOINT", config.CHAT_COMPLETIONS_ENDPOINT)
                url = f"{client.base_url}{endpoint}"
                response = await client.http_client.post(
                    url,
                    headers=headers,
                    json=data,
                )

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise NetworkError(
                        err.NETWORK_ERROR.format(
                            error=f"HTTP {e.response.status_code}: {e.response.text}"
                        )
                    ) from e

                if stream:
                    return AsyncStream(response.aiter_lines())

                response_data = response.json()
                default_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }

                # Handle Ollama response format
                if "message" in response_data:
                    message = response_data.get("message", {}).get("content", "")
                    return ChatCompletion(
                        id=f"ollama-{hash(str(response_data))}",
                        object="chat.completion",
                        created=int(response_data.get("created", 0)),
                        model=model,
                        choices=[
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": message},
                                "finish_reason": "stop",
                            }
                        ],
                        usage=default_usage,
                    )

                # Handle standard chat completion format
                return ChatCompletion(
                    id=response_data.get("id", ""),
                    object=response_data.get("object", "chat.completion"),
                    created=response_data.get("created", 0),
                    model=model,
                    choices=response_data.get("choices", []),
                    usage=response_data.get("usage", default_usage),
                )


########################################################
#           Public Methods
########################################################


def setup_client() -> DeepSeekClient:
    """Set up and configure the DeepSeek client with optimal settings."""
    client = DeepSeekClient()
    set_default_model_client(client)
    return client
