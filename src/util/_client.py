from __future__ import annotations

import os
from typing import Any

import httpx

from ..models.shared import set_default_model_client
from ._constants import (
    CHAT_COMPLETIONS_ENDPOINT,
    DEFAULT_BASE_URL,
    HEADERS,
    HTTP_MAX_CONNECTIONS,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_TIMEOUT_CONNECT,
    HTTP_TIMEOUT_READ,
    HTTP_TIMEOUT_TOTAL,
)
from ._types import (
    AsyncDeepSeek,
    AsyncStream,
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
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
        self.api_key = api_key
        self.base_url = (base_url or os.getenv("BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
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

                if temperature is not None:
                    data["temperature"] = temperature
                if top_p is not None:
                    data["top_p"] = top_p
                if max_tokens is not None:
                    data["max_tokens"] = max_tokens

                endpoint = os.getenv("CHAT_COMPLETIONS_ENDPOINT", CHAT_COMPLETIONS_ENDPOINT)
                url = f"{client.base_url}{endpoint}"
                response = await client.http_client.post(
                    url,
                    headers=headers,
                    json=data,
                )

                response.raise_for_status()

                if stream:
                    return AsyncStream(response.aiter_lines())

                # Convert Ollama response to ChatCompletion format
                ollama_response = response.json()
                message = ollama_response.get("message", {}).get("content", "")
                return ChatCompletion(
                    id=f"ollama-{hash(str(ollama_response))}",
                    object="chat.completion",
                    created=int(ollama_response.get("created", 0)),
                    model=model,
                    choices=[
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": message},
                            "finish_reason": "stop",
                        }
                    ],
                    usage={
                        "prompt_tokens": 0,  # Ollama doesn't provide token counts
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                )


########################################################
#           Public Methods
########################################################


def setup_client() -> DeepSeekClient:
    """Set up and configure the DeepSeek client with optimal settings."""
    client = DeepSeekClient(
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(
                HTTP_TIMEOUT_TOTAL, connect=HTTP_TIMEOUT_CONNECT, read=HTTP_TIMEOUT_READ
            ),
            limits=httpx.Limits(
                max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
                max_connections=HTTP_MAX_CONNECTIONS,
            ),
        )
    )
    set_default_model_client(client)
    return client
