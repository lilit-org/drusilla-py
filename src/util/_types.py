from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypedDict, Union, cast

from typing_extensions import TypeVar

from ._constants import NOT_GIVEN

########################################################
#              Type Variables                         #
########################################################

T = TypeVar("T")
MaybeAwaitable = Union[Awaitable[T], T]

########################################################
#            Data class for types                      #
########################################################

@dataclass
class Usage:
    """Token usage statistics for API calls."""
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclass
class ResponseCompletedEvent:
    """Event indicating completion of a response."""
    response: Response

@dataclass
class ResponseContentPartAddedEvent:
    """Event indicating a new content part has been added to a response."""
    content_index: int
    item_id: str
    output_index: int
    part: ResponseOutput
    type: Literal["response.content_part.added"] = "response.content_part.added"

@dataclass
class ResponseContentPartDoneEvent:
    """Event indicating a content part has been completed."""
    content_index: int
    item_id: str
    output_index: int
    part: ResponseOutput
    type: Literal["response.content_part.done"] = "response.content_part.done"

@dataclass
class Response:
    """API response containing outputs and usage stats."""
    id: str
    output: list[ResponseOutput]
    usage: Usage | None = None
    created_at: float | None = None
    model: str | None = None
    object: str | None = None
    tool_choice: str | None = None
    temperature: float | None = None
    tools: list[Any] | None = None
    parallel_tool_calls: bool | None = None
    top_p: float | None = None


########################################################
#            TypedDict for Response Outputs            #
########################################################

class ComputerAction(TypedDict):
    """Computer interaction actions like clicks, typing, etc."""
    type: Literal["click", "double_click", "drag", "keypress", "move", "screenshot", "scroll", "type", "wait"]
    x: int | None
    y: int | None
    button: str | None
    keys: list[str] | None
    path: list[dict[str, int]] | None
    scroll_x: int | None
    scroll_y: int | None
    text: str | None


class ResponseOutput(TypedDict):
    """Output from an API response with optional content and metadata."""
    type: str
    content: str | None
    name: str | None
    arguments: dict[str, Any] | None
    call_id: str | None
    action: ComputerAction | None


class ResponseOutputText(TypedDict):
    """Text content in a response output."""
    type: Literal["output_text"]
    text: str
    annotations: list[Any]


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
    content_index: int | None
    item_id: str | None
    output_index: int | None
    delta: str | None
    part: ResponseOutput | None
    response: Response | None


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


class ComputerCallOutput(TypedDict):
    """Output from a computer tool call."""
    type: Literal["computer_call_output"]
    call_id: str
    output: str


ResponseOutputItem = ResponseOutput
ResponseOutputMessage = ResponseOutput
ResponseFileSearchToolCall = ResponseOutput
ResponseFunctionWebSearch = ResponseOutput
ResponseComputerToolCall = ResponseOutput
ResponseReasoningItem = ResponseOutput


########################################################
#            TypedDict for Input Items                 #
########################################################

class ResponseInputItemParam(TypedDict):
    """Input item for API requests."""
    type: str
    content: str | None
    role: Literal["user", "assistant", "system", "developer"] | None
    name: str | None
    arguments: dict[str, Any] | None
    call_id: str | None
    action: ComputerAction | None


########################################################
#            TypedDict for Chat Completion            #
########################################################

class ChatCompletionFunctionParam(TypedDict):
    """Parameters for a function call in chat completion."""
    name: str
    description: str
    parameters: dict[str, Any]

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
    text: str | None
    image_url: dict[str, str] | None

class ChatCompletionMessage(TypedDict):
    """Message in a chat completion."""
    role: Literal["user", "assistant", "system", "developer", "tool"]
    content: str | None
    tool_calls: list[ChatCompletionMessageToolCallParam] | None
    refusal: str | None
    audio: Any | None

class ChatCompletionMessageParam(TypedDict):
    """Parameters for a chat completion message."""
    role: Literal["user", "assistant", "system", "developer", "tool"]
    content: str | list[ChatCompletionContentPartParam]
    tool_call_id: str | None
    tool_calls: list[ChatCompletionMessageToolCallParam] | None
    refusal: str | None

class ChatCompletionUsage(TypedDict):
    """Token usage statistics for chat completion."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionDeltaFunction(TypedDict):
    """Delta update for a function call in streaming responses."""
    name: str | None
    arguments: str | None

class ChatCompletionDeltaToolCall(TypedDict):
    """Delta update for a tool call in streaming responses."""
    index: int
    id: str | None
    type: Literal["function"] | None
    function: ChatCompletionDeltaFunction | None

class ChatCompletionDelta(TypedDict):
    """Delta update in streaming responses."""
    content: str | None
    tool_calls: list[ChatCompletionDeltaToolCall] | None

class ChatCompletionChoice(TypedDict):
    """Choice in a chat completion response."""
    index: int
    message: ChatCompletionMessage
    finish_reason: str | None
    delta: ChatCompletionDelta | None

class ChatCompletion(TypedDict):
    """Complete chat completion response."""
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None

class ChatCompletionChunk(TypedDict):
    """Chunk in a streaming chat completion response."""
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None


ChatCompletionToolChoiceOptionParam: TypeAlias = Union[Literal["auto", "required", "none"], dict[str, Any]]


########################################################
#            TypedDict for Response Format             #
########################################################

class ResponseFormat(TypedDict):
    """Format specification for API responses."""
    type: Literal["json_schema"]
    json_schema: dict[str, Any]


########################################################
#            Async DeepSeek Client                    #
########################################################

class AsyncStream(AsyncIterator[ChatCompletionChunk]):
    """Async iterator for streaming chat completion chunks."""
    async def __anext__(self) -> ChatCompletionChunk:
        raise NotImplementedError

class AsyncDeepSeek:
    """Async client for DeepSeek API interactions."""
    class chat:
        class completions:
            @classmethod
            async def create(
                cls,
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
                raise NotImplementedError
