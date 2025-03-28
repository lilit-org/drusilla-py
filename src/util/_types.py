from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from typing_extensions import TypeVar

########################################################
#              Type Variables
########################################################

T = TypeVar("T")
MaybeAwaitable = Awaitable[T] | T

########################################################
#            Data class for Usage
########################################################


@dataclass(frozen=True)
class Usage:
    """Token usage statistics for API calls."""

    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


########################################################
#            TypedDict for Computer Actions
########################################################


class ComputerAction(TypedDict):
    """Computer interaction actions like clicks, typing, etc."""

    type: Literal[
        "click",
        "double_click",
        "drag",
        "keypress",
        "move",
        "screenshot",
        "scroll",
        "type",
        "wait",
    ]
    x: NotRequired[int]
    y: NotRequired[int]
    button: NotRequired[str]
    keys: NotRequired[Sequence[str]]
    path: NotRequired[Sequence[Mapping[str, int]]]
    scroll_x: NotRequired[int]
    scroll_y: NotRequired[int]
    text: NotRequired[str]


class ComputerCallOutput(TypedDict):
    """Output from a computer tool call."""

    type: Literal["computer_call_output"]
    call_id: str
    output: str


########################################################
#            Main classes for Response Outputs
########################################################


@dataclass(frozen=True)
class ResponseCompletedEvent:
    """Event indicating completion of a response."""

    response: Response


@dataclass(frozen=True)
class ResponseContentPartAddedEvent:
    """Event indicating a new content part has been added to a response."""

    content_index: int
    item_id: str
    output_index: int
    part: ResponseOutput
    type: Literal["response.content_part.added"] = "response.content_part.added"


@dataclass(frozen=True)
class ResponseContentPartDoneEvent:
    """Event indicating a content part has been completed."""

    content_index: int
    item_id: str
    output_index: int
    part: ResponseOutput
    type: Literal["response.content_part.done"] = "response.content_part.done"


@dataclass(frozen=True)
class Response:
    """API response containing outputs and usage stats."""

    id: str
    output: Sequence[ResponseOutput]
    usage: Usage | None = None
    created_at: float | None = None
    model: str | None = None
    object: Literal["response"] | None = None
    tool_choice: Literal["auto", "required", "none"] | None = None
    temperature: float | None = None
    tools: Sequence[ChatCompletionToolParam] | None = None
    parallel_tool_calls: bool | None = None
    top_p: float | None = None


class ResponseOutput(TypedDict):
    """Output from an API response with optional content and metadata."""

    type: str
    content: NotRequired[str | Sequence[ResponseOutputText | ResponseOutputRefusal]]
    name: NotRequired[str]
    arguments: NotRequired[Mapping[str, Any]]
    call_id: NotRequired[str]
    action: NotRequired[ComputerAction]
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


class ResponseFunctionToolCall(TypedDict):
    """Function tool call in a response output."""

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
    """Output from a function tool call."""

    type: Literal["function_call_output"]
    call_id: str
    output: str


@dataclass(frozen=True)
class ResponseEvent:
    """Event indicating a change in response state."""

    type: Literal[
        "completed", "content_part.added", "content_part.done", "output_text.delta"
    ]
    response: Response | None = None
    content_index: int | None = None
    item_id: str | None = None
    output_index: int | None = None
    part: ResponseOutput | None = None
    delta: str | None = None


########################################################
#           Main classes for Chat Completion
########################################################


class ChatCompletionFunctionParam(TypedDict):
    """Parameters for a function call in chat completion."""

    name: str
    description: str
    parameters: Mapping[str, Any]


class ChatCompletionToolParam(TypedDict):
    """Tool parameters for chat completion."""

    type: Literal["function"]
    function: ChatCompletionFunctionParam


class ChatCompletionMessageToolCallParam(TypedDict):
    """Tool call parameters in a chat message."""

    id: str
    type: Literal["function"]
    function: ChatCompletionFunctionParam


class ChatCompletionContentPartParam(TypedDict):
    """Content part parameters for chat completion messages."""

    type: Literal["text", "image_url"]
    text: NotRequired[str]
    image_url: NotRequired[Mapping[str, str]]


class ChatCompletionMessage(TypedDict):
    """Message in a chat completion."""

    role: Literal["user", "assistant", "system", "developer", "tool"]
    content: NotRequired[str]
    tool_calls: NotRequired[Sequence[ChatCompletionMessageToolCallParam]]
    refusal: NotRequired[str]
    audio: NotRequired[Mapping[str, str]]


class ChatCompletionMessageParam(TypedDict):
    """Parameters for a chat completion message."""

    role: Literal["user", "assistant", "system", "developer", "tool"]
    content: str | Sequence[ChatCompletionContentPartParam]
    tool_call_id: NotRequired[str]
    tool_calls: NotRequired[Sequence[ChatCompletionMessageToolCallParam]]
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


class ChatCompletionDeltaToolCall(TypedDict):
    """Delta update for a tool call in streaming responses."""

    index: int
    id: NotRequired[str]
    type: NotRequired[Literal["function"]]
    function: NotRequired[ChatCompletionDeltaFunction]


class ChatCompletionDelta(TypedDict):
    """Delta update in streaming responses."""

    content: NotRequired[str]
    tool_calls: NotRequired[Sequence[ChatCompletionDeltaToolCall]]


class ChatCompletionChoice(TypedDict):
    """Choice in a chat completion response."""

    index: int
    message: ChatCompletionMessage
    finish_reason: NotRequired[str]
    delta: NotRequired[ChatCompletionDelta]


class ChatCompletion(TypedDict):
    """Complete chat completion response."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: Sequence[ChatCompletionChoice]
    usage: NotRequired[ChatCompletionUsage]


class ChatCompletionChunk(TypedDict):
    """Chunk in a streaming chat completion response."""

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: Sequence[ChatCompletionChoice]
    usage: NotRequired[ChatCompletionUsage]


ChatCompletionToolChoiceOptionParam: TypeAlias = (
    Literal["auto", "required", "none"] | Mapping[str, Any]
)


########################################################
#            TypedDict for Response Input
########################################################


class ResponseInputItemParam(TypedDict):
    """Input item for API requests."""

    type: str
    content: NotRequired[str]
    role: NotRequired[Literal["user", "assistant", "system", "developer"]]
    name: NotRequired[str]
    arguments: NotRequired[Mapping[str, Any]]
    call_id: NotRequired[str]
    action: NotRequired[ComputerAction]


class ResponseFormat(TypedDict):
    """Format specification for API responses."""

    type: Literal["json_schema"]
    json_schema: Mapping[str, Any]


########################################################
#           Main class: AsyncStream
########################################################


class AsyncStream(AsyncIterator[ChatCompletionChunk]):
    """Async iterator for streaming chat completion chunks."""

    def __init__(self, stream: AsyncIterator[str]):
        self._stream = stream

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            line = await anext(self._stream)
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
    """Async client for DeepSeek API interactions."""

    class chat:
        class completions:
            @classmethod
            async def create(
                cls,
                model: str,
                messages: Sequence[ChatCompletionMessageParam],
                tools: Sequence[ChatCompletionToolParam] | None = None,
                temperature: float | None = None,
                top_p: float | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                max_tokens: int | None = None,
                tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
                response_format: ResponseFormat | None = None,
                parallel_tool_calls: bool | None = None,
                stream: bool = False,
                stream_options: Mapping[str, bool] | None = None,
                extra_headers: Mapping[str, str] | None = None,
            ) -> ChatCompletion | AsyncStream:
                """Create a chat completion with the given parameters."""
                raise NotImplementedError


ResponseOutputItem: TypeAlias = ResponseOutput
ResponseOutputMessage: TypeAlias = ResponseOutput
ResponseFileSearchToolCall: TypeAlias = ResponseOutput
ResponseFunctionWebSearch: TypeAlias = ResponseOutput
ResponseComputerToolCall: TypeAlias = ResponseOutput
ResponseReasoningItem: TypeAlias = ResponseOutput
