"""
This module defines core type definitions and data structures used throughout the project.

It includes:
- Type variables and generic type definitions
- Data classes for tracking API usage and context management
- Response output types for API interactions
- Chat completion related types for LLM interactions
- Streaming response handling types
- Input parameter types for API requests

These types provide a consistent interface for working with LLM APIs, handling responses,
and managing streaming data across the application.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeAlias, TypedDict

from typing_extensions import NotRequired, TypeVar

########################################################
#              Type Variables
########################################################

TContext = TypeVar("TContext", default=Any)
T = TypeVar("T")
MaybeAwaitable = Awaitable[T] | T


########################################################
#            Queue Sentinel Types
########################################################


@dataclass(frozen=True)
class QueueCompleteSentinel:
    """Sentinel value used to indicate the end of a queue stream."""


########################################################
#            Data class for Usage and Contexts
########################################################


@dataclass(frozen=True)
class Usage:
    """Track LLM API token usage and requests."""

    requests: int = 0
    """API request count."""

    input_tokens: int = 0
    """Tokens sent to API."""

    output_tokens: int = 0
    """Tokens received from API."""

    total_tokens: int = 0
    """Total tokens used."""

    def add(self, other: Usage) -> Usage:
        return Usage(
            requests=self.requests + (other.requests or 0),
            input_tokens=self.input_tokens + (other.input_tokens or 0),
            output_tokens=self.output_tokens + (other.output_tokens or 0),
            total_tokens=self.total_tokens + (other.total_tokens or 0),
        )


@dataclass
class RunContextWrapper(Generic[TContext]):
    """Wrapper for context objects passed to Runner.run().

    Contexts are used to pass dependencies and data to custom code.
    They are not passed to the LLM.
    """

    context: TContext
    """Context object passed to Runner.run()"""

    usage: Usage = field(default_factory=Usage)
    """Usage stats for the agent run. May be stale during streaming until final chunk."""


########################################################
#            Classes for Response Outputs
########################################################


@dataclass(frozen=True)
class Response:
    """API response containing outputs and usage stats."""

    id: str
    output: Sequence[ResponseOutput]
    usage: Usage | None = None
    created_at: float | None = None
    model: str | None = None
    object: Literal["response"] | None = None
    sword_choice: Literal["auto", "required", "none"] | None = None
    temperature: float | None = None
    swords: Sequence[ChatCompletionSwordParam] | None = None
    parallel_sword_calls: bool | None = None
    top_p: float | None = None


class ResponseOutput(TypedDict):
    """Output from an API response with optional content and metadata."""

    type: str
    content: NotRequired[str | Sequence[ResponseOutputText | ResponseOutputRefusal]]
    name: NotRequired[str]
    arguments: NotRequired[Mapping[str, Any]]
    call_id: NotRequired[str]
    role: NotRequired[Literal["user", "assistant", "system", "developer"]]
    status: NotRequired[str]


class ResponseOutputText(TypedDict):
    """Text content in a response output."""

    type: Literal["output_text"]
    text: str
    annotations: Sequence[Mapping[str, Any]]


class ResponseOutputRefusal(TypedDict):
    """Refusal content in a response output."""

    type: Literal["refusal"]
    refusal: str


class ResponseFunctionSwordCall(TypedDict):
    """Function sword call in a response output."""

    type: Literal["function_call"]
    id: str
    call_id: str
    name: str
    arguments: str


class ResponseStreamEvent(TypedDict):
    """Event in a streaming response."""

    type: str
    content_index: NotRequired[int]
    item_id: NotRequired[str]
    output_index: NotRequired[int]
    delta: NotRequired[str]
    part: NotRequired[ResponseOutput]
    response: NotRequired[Response]


class ResponseTextDeltaEvent(TypedDict):
    """Event for text delta updates in streaming responses."""

    type: Literal["response.output_text.delta"]
    content_index: int
    item_id: str
    output_index: int
    delta: str


class FunctionCallOutput(TypedDict):
    """Output from a function sword call."""

    type: Literal["function_call_output"]
    call_id: str
    output: str


@dataclass(frozen=True)
class ResponseEvent:
    """Event indicating a change in response state."""

    type: Literal["completed", "content_part.added", "content_part.done", "output_text.delta"]
    response: Response | None = None
    content_index: int | None = None
    item_id: str | None = None
    output_index: int | None = None
    part: ResponseOutput | None = None
    delta: str | None = None


########################################################
#           Classes for Chat Completion Types
########################################################


class ChatCompletionSwordParam(TypedDict):
    """Sword params for chat completion."""

    name: str
    description: str
    parameters: Mapping[str, Any]


class ChatCompletionMessageSwordCallParam(TypedDict):
    """Sword call parameters in a chat message."""

    id: str
    type: Literal["function"]
    function: ChatCompletionSwordParam


class ChatCompletionContentPartParam(TypedDict):
    """Content part parameters for chat completion messages."""

    type: Literal["text", "image_url"]
    text: NotRequired[str]
    image_url: NotRequired[Mapping[str, str]]


class ChatCompletionMessage(TypedDict):
    """Message in a chat completion."""

    role: Literal["user", "assistant", "system", "developer", "sword"]
    content: NotRequired[str]
    sword_calls: NotRequired[Sequence[ChatCompletionMessageSwordCallParam]]
    refusal: NotRequired[str]
    audio: NotRequired[Mapping[str, str]]


class ChatCompletionMessageParam(TypedDict):
    """Parameters for a chat completion message."""

    role: Literal["user", "assistant", "system", "developer", "sword"]
    content: str | Sequence[ChatCompletionContentPartParam]
    sword_call_id: NotRequired[str]
    sword_calls: NotRequired[Sequence[ChatCompletionMessageSwordCallParam]]
    refusal: NotRequired[str]


class ChatCompletionUsage(TypedDict):
    """Token usage statistics for chat completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionDeltaFunction(TypedDict):
    """Delta update for a function call in streaming responses."""

    name: NotRequired[str]
    arguments: NotRequired[str]


class ChatCompletionDeltaSwordCall(TypedDict):
    """Delta update for a sword call in streaming responses."""

    index: int
    id: NotRequired[str]
    type: NotRequired[Literal["function"]]
    function: NotRequired[ChatCompletionDeltaFunction]


class ChatCompletionDelta(TypedDict):
    """Delta update for streaming responses."""

    role: NotRequired[Literal["assistant"]]
    content: NotRequired[str]
    function: NotRequired[dict[str, str]]


class ChatCompletionChoice(TypedDict):
    """Choice in a chat completion response."""

    index: int
    message: ChatCompletionMessage
    finish_reason: NotRequired[str]
    delta: NotRequired[ChatCompletionDelta]


class ChatCompletion(TypedDict):
    """Chat completion response."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: Sequence[ChatCompletionChoice]
    usage: NotRequired[ChatCompletionUsage]


class ChatCompletionChunk(TypedDict):
    """Streaming response chunk."""

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: Sequence[ChatCompletionChoice]
    usage: NotRequired[ChatCompletionUsage]


ChatCompletionSwordChoiceOptionParam: TypeAlias = (
    Literal["auto", "required", "none"] | Mapping[str, Any]
)


########################################################
#            Class for Response Input
########################################################


class ResponseInputItemParam(TypedDict):
    """Input item for API requests."""

    type: str
    content: NotRequired[str]
    role: NotRequired[Literal["user", "assistant", "system", "developer"]]
    name: NotRequired[str]
    arguments: NotRequired[Mapping[str, Any]]
    call_id: NotRequired[str]


class ResponseFormat(TypedDict):
    """Format specification for API responses."""

    type: Literal["json_schema"]
    json_schema: Mapping[str, Any]


########################################################
#           Main class for Async Streaming
########################################################


class AsyncStream(AsyncIterator[ChatCompletionChunk]):
    """Async iterator for streaming chat completion chunks."""

    def __init__(self, stream: AsyncIterator[str | dict]):
        self._stream = stream

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            line = await anext(self._stream)

            # If line is already a dictionary, return it directly
            if isinstance(line, dict):
                return line

            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                raise StopAsyncIteration
            import json

            data = json.loads(line)

            # Handle Ollama's specific response format
            if "message" in data:
                content = data["message"].get("content", "")
                data = {
                    "id": data.get("id", "ollama-" + str(hash(content))),
                    "object": "chat.completion.chunk",
                    "created": data.get("created", int(time.time())),
                    "model": data.get("model", "unknown"),
                    "choices": [{"index": 0, "delta": {"content": content}}],
                }
            # If the response doesn't match the expected structure, create a valid chunk
            elif not isinstance(data, dict):
                data = {
                    "id": "fallback-id",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "unknown",
                    "choices": [{"index": 0, "delta": {"content": str(data)}}],
                }
            elif "choices" not in data:
                data = {
                    "id": data.get("id", "fallback-id"),
                    "object": "chat.completion.chunk",
                    "created": data.get("created", int(time.time())),
                    "model": data.get("model", "unknown"),
                    "choices": [{"index": 0, "delta": {"content": str(data)}}],
                }
            elif not data["choices"]:
                data["choices"] = [{"index": 0, "delta": {"content": ""}}]
            elif "delta" not in data["choices"][0]:
                data["choices"][0]["delta"] = {"content": str(data["choices"][0])}

            return data
        except StopAsyncIteration:
            raise
        except Exception as e:
            raise RuntimeError(f"Error parsing stream chunk: {e}") from e


class AsyncDeepSeek:
    """Async DeepSeek API client."""

    class chat:
        class completions:
            @classmethod
            async def create(
                cls,
                model: str,
                messages: Sequence[ChatCompletionMessageParam],
                swords: Sequence[ChatCompletionSwordParam] | None = None,
                temperature: float | None = None,
                top_p: float | None = None,
                max_tokens: int | None = None,
                sword_choice: ChatCompletionSwordChoiceOptionParam | None = None,
                response_format: ResponseFormat | None = None,
                parallel_sword_calls: bool | None = None,
                stream: bool = False,
                stream_options: Mapping[str, bool] | None = None,
                extra_headers: Mapping[str, str] | None = None,
            ) -> ChatCompletion | AsyncStream:
                """Create chat completion."""
                raise NotImplementedError


ResponseReasoningItem: TypeAlias = ResponseOutput
