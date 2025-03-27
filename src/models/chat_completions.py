from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ..agents.output import AgentOutputSchema
from ..util._constants import FAKE_RESPONSES_ID, HEADERS, UNSET
from ..util._exceptions import AgentError, UsageError
from ..util._handoffs import Handoff
from ..util._items import (
    ModelResponse,
    TResponseInputItem,
    TResponseOutputItem,
    TResponseStreamEvent,
)
from ..util._logger import logger
from ..util._tool import FunctionTool, Tool
from ..util._types import (
    AsyncDeepSeek,
    AsyncStream,
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFormat,
    ResponseFunctionToolCall,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)
from ..util._usage import Usage
from .interface import Model
from .settings import ModelSettings

########################################################
#           Data Classes                               #
########################################################

@dataclass(frozen=True)
class _StreamingState:
    """Maintains the current state of streaming responses."""
    text_content_index_and_output: tuple[int, ResponseOutputText] | None = None
    function_calls: dict[int, ResponseFunctionToolCall] = field(default_factory=dict)


########################################################
#           Main Class: Chat Completions Model         #
########################################################

class ModelChatCompletionsModel(Model):
    """Handles chat completion requests to the API model."""

    def __init__(
        self,
        model: str,
        model_client: AsyncDeepSeek,
    ) -> None:
        self.model = model
        self._client = model_client

    def _non_null_or_not_given(self, value: Any) -> Any:
        """Converts falsy values to None, preserving non-falsy values."""
        return None if not value else value

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
    ) -> ModelResponse:
        """Retrieves a complete model response."""
        response = await self._fetch_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            stream=False,
        )

        logger.debug("\n✅ Successfully received model response...")
        if isinstance(response, tuple):
            response_obj = response[0]
        else:
            response_obj = response

        usage = (
            Usage(
                requests=1,
                input_tokens=response_obj['usage']['prompt_tokens'],
                output_tokens=response_obj['usage']['completion_tokens'],
                total_tokens=response_obj['usage']['total_tokens'],
            )
            if response_obj.get('usage')
            else Usage()
        )

        items = _Converter.message_to_output_items(response_obj['choices'][0]['message'])

        return ModelResponse(
            output=items,
            usage=usage,
            referenceable_id=None,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Stream model responses as generated."""
        _, stream = await self._fetch_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            stream=True,
        )

        state = _StreamingState()
        async for chunk in stream:
            if chunk["choices"][0]["delta"]["content"]:
                delta = chunk["choices"][0]["delta"]
                if not state.text_content_index_and_output:
                    state.text_content_index_and_output = (0, ResponseOutputText(
                        text=delta["content"],
                        type="output_text",
                        annotations=[],
                    ))
                    yield ResponseContentPartAddedEvent(
                        content_index=state.text_content_index_and_output[0],
                        item_id=FAKE_RESPONSES_ID,
                        output_index=0,
                        part=ResponseOutputText(
                            text=delta["content"],
                            type="output_text",
                            annotations=[],
                        ),
                        type="response.content_part.added",
                    )
                else:
                    state.text_content_index_and_output[1].text += delta["content"]
                    yield ResponseTextDeltaEvent(
                        content_index=state.text_content_index_and_output[0],
                        delta=delta["content"],
                        item_id=FAKE_RESPONSES_ID,
                        output_index=0,
                        type="response.output_text.delta",
                    )

            if chunk["choices"][0]["delta"]["tool_calls"]:
                for tc_delta in chunk["choices"][0]["delta"]["tool_calls"]:
                    if tc_delta["index"] not in state.function_calls:
                        state.function_calls[tc_delta["index"]] = ResponseFunctionToolCall(
                            id=FAKE_RESPONSES_ID,
                            arguments="",
                            name="",
                            type="function_call",
                            call_id="",
                        )
                    tc_function = tc_delta["function"]

                    state.function_calls[tc_delta["index"]].arguments += (
                        tc_function["arguments"] if tc_function else ""
                    ) or ""
                    state.function_calls[tc_delta["index"]].name += (
                        tc_function["name"] if tc_function else ""
                    ) or ""
                    state.function_calls[tc_delta["index"]].call_id += tc_delta["id"] or ""

            if chunk["choices"][0]["finish_reason"] == "stop":
                if state.text_content_index_and_output:
                    yield ResponseContentPartDoneEvent(
                        content_index=state.text_content_index_and_output[0],
                        item_id=FAKE_RESPONSES_ID,
                        output_index=0,
                        part=state.text_content_index_and_output[1],
                        type="response.content_part.done",
                    )

                for i, function_call in state.function_calls.items():
                    yield ResponseContentPartAddedEvent(
                        content_index=i + (1 if state.text_content_index_and_output else 0),
                        item_id=FAKE_RESPONSES_ID,
                        output_index=0,
                        part=function_call,
                        type="response.content_part.added",
                    )
                    yield ResponseContentPartDoneEvent(
                        content_index=i + (1 if state.text_content_index_and_output else 0),
                        item_id=FAKE_RESPONSES_ID,
                        output_index=0,
                        part=function_call,
                        type="response.content_part.done",
                    )

                final_response = Response(
                    id=FAKE_RESPONSES_ID,
                    created_at=time.time(),
                    model=self.model,
                    object="response",
                    output=[],
                    tool_choice="auto",
                    top_p=model_settings.top_p,
                    temperature=model_settings.temperature,
                    tools=[],
                    parallel_tool_calls=False,
                )

                if state.text_content_index_and_output:
                    final_response.output.append(
                        {
                            "id": FAKE_RESPONSES_ID,
                            "content": [state.text_content_index_and_output[1]],
                            "role": "assistant",
                            "type": "message",
                            "status": "completed",
                        }
                    )

                for function_call in state.function_calls.values():
                    final_response.output.append(function_call)

                yield ResponseCompletedEvent(
                    response=final_response,
                    type="response.completed",
                )

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        stream: bool = False,
    ) -> ChatCompletion | tuple[Response, AsyncStream]:
        """Makes a request to the chat completions API and returns the response."""
        converted_messages = _Converter.items_to_messages(input)

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )

        parallel_tool_calls = (
            True if model_settings.parallel_tool_calls and tools and len(tools) > 0 else UNSET
        )
        tool_choice = _Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = _Converter.convert_response_format(output_schema)
        converted_tools = [ToolConverter.to_api_format(tool) for tool in tools] if tools else []

        for handoff in handoffs:
            converted_tools.append(ToolConverter.convert_handoff_tool(handoff))

        request_params = {
            "model": self.model,
            "messages": converted_messages,
            "temperature": self._non_null_or_not_given(model_settings.temperature),
            "top_p": self._non_null_or_not_given(model_settings.top_p),
            "frequency_penalty": self._non_null_or_not_given(model_settings.frequency_penalty),
            "presence_penalty": self._non_null_or_not_given(model_settings.presence_penalty),
            "max_tokens": self._non_null_or_not_given(model_settings.max_tokens),
            "stream": stream,
            "extra_headers": HEADERS,
        }

        if converted_tools:
            request_params["tools"] = converted_tools
        if tool_choice != UNSET:
            request_params["tool_choice"] = tool_choice
        if response_format != UNSET:
            request_params["response_format"] = response_format
        if parallel_tool_calls != UNSET:
            request_params["parallel_tool_calls"] = parallel_tool_calls
        if stream:
            request_params["stream_options"] = {"include_usage": True}

        ret = await self._get_client().chat.completions.create(**request_params)

        if stream:
            response = Response(
                id=FAKE_RESPONSES_ID,
                created_at=time.time(),
                model=self.model,
                object="response",
                output=[],
                tool_choice=cast(Literal["auto", "required", "none"], tool_choice)
                if tool_choice != UNSET
                else "auto",
                top_p=model_settings.top_p,
                temperature=model_settings.temperature,
                tools=[],
                parallel_tool_calls=parallel_tool_calls or False,
            )
            return response, ret

        return ret

    def _get_client(self) -> AsyncDeepSeek:
        """Returns the configured API client instance."""
        return self._client


########################################################
#           Main Class: Converter                      #
########################################################

class _Converter:
    """Converts between API and internal data formats."""

    @classmethod
    def convert_tool_choice(
        cls, tool_choice: Literal["auto", "required", "none"] | str | None
    ) -> ChatCompletionToolChoiceOptionParam | Any:
        """Converts tool choice settings to API format."""
        if tool_choice is None:
            return UNSET
        elif tool_choice == "auto":
            return "auto"
        elif tool_choice == "required":
            return "required"
        elif tool_choice == "none":
            return "none"
        else:
            return {
                "type": "function",
                "function": {
                    "name": tool_choice,
                },
            }

    @classmethod
    def convert_response_format(
        cls, final_output_schema: AgentOutputSchema | None
    ) -> ResponseFormat | Any:
        """Converts output schema to API response format."""
        if not final_output_schema or final_output_schema.is_plain_text():
            return UNSET
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "final_output",
                "strict": final_output_schema.strict_json_schema,
                "schema": final_output_schema.json_schema(),
            },
        }

    @classmethod
    def message_to_output_items(cls, message: ChatCompletionMessage | dict[str, Any]) -> list[TResponseOutputItem]:
        """Convert message to output items."""
        items: list[TResponseOutputItem] = []
        message_dict = message if isinstance(message, dict) else message.__dict__

        if message_dict.get('content'):
            items.append(
                ResponseOutputText(
                    text=message_dict['content'],
                    type="output_text",
                    annotations=[],
                )
            )
        if message_dict.get('tool_calls'):
            for tool_call in message_dict['tool_calls']:
                items.append(
                    ResponseFunctionToolCall(
                        name=tool_call['function']['name'],
                        arguments=tool_call['function']['arguments'],
                        type="function_call",
                        call_id=tool_call.get('id', ''),
                        referenceable_id=None,
                    )
                )

        return items

    @classmethod
    def maybe_easy_input_message(cls, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return

        keys = item.keys()
        if keys != {"content", "role"}:
            return

        role = item.get("role", None)
        if role not in ("user", "assistant", "system", "developer"):
            return

        if "content" not in item:
            return

        return cast(dict[str, Any], item)

    @classmethod
    def maybe_input_message(cls, item: Any) -> dict[str, Any] | None:
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role")
            in (
                "user",
                "system",
                "developer",
            )
        ):
            return cast(dict[str, Any], item)

    @classmethod
    def maybe_file_search_call(cls, item: Any) -> dict[str, Any] | None:
        if isinstance(item, dict) and item.get("type") == "file_search_call":
            return cast(dict[str, Any], item)

    @classmethod
    def maybe_function_tool_call(cls, item: Any) -> dict[str, Any] | None:
        if isinstance(item, dict) and item.get("type") == "function_call":
            return cast(dict[str, Any], item)

    @classmethod
    def maybe_function_tool_call_output(
        cls,
        item: Any,
    ) -> dict[str, Any] | None:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            return cast(dict[str, Any], item)

    @classmethod
    def maybe_item_reference(cls, item: Any) -> dict[str, Any] | None:
        if isinstance(item, dict) and item.get("type") == "item_reference":
            return cast(dict[str, Any], item)

    @classmethod
    def maybe_response_output_message(cls, item: Any) -> dict[str, Any] | None:
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "assistant"
        ):
            return cast(dict[str, Any], item)

    @classmethod
    def extract_text_content(
        cls, content: str | Iterable[dict[str, Any]]
    ) -> str | list[dict[str, Any]]:
        all_content = cls.extract_all_content(content)
        if isinstance(all_content, str):
            return all_content
        return [c for c in all_content if c.get("type") == "text"]

    @classmethod
    def extract_all_content(
        cls, content: str | Iterable[dict[str, Any]]
    ) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content

        out: list[dict[str, Any]] = []
        for c in content:
            if not isinstance(c, dict):
                continue

            if c.get("type") == "input_text":
                out.append({
                    "type": "text",
                    "text": c["text"],
                })
            elif c.get("type") == "input_image":
                if "image_url" not in c or not c["image_url"]:
                    raise AgentError(f"Only image URLs are supported for input_image {c}")
                out.append({
                    "type": "image_url",
                    "image_url": {
                        "url": c["image_url"],
                        "detail": c["detail"],
                    },
                })
            elif c.get("type") == "input_file":
                raise UsageError(f"File uploads are not supported for chat completions {c}")
            else:
                raise UsageError(f"Unknown content: {c}")
        return out

    @classmethod
    def items_to_messages(
        cls,
        items: str | Iterable[TResponseInputItem],
    ) -> list[ChatCompletionMessageParam]:
        """
        Convert a sequence of 'Item' objects into a list of ChatCompletionMessageParam.

        Rules:
        - EasyInputMessage or InputMessage (role=user) => ChatCompletionUserMessageParam
        - EasyInputMessage or InputMessage (role=system) => ChatCompletionSystemMessageParam
        - EasyInputMessage or InputMessage (role=developer) => ChatCompletionDeveloperMessageParam
        - InputMessage (role=assistant) => Start or flush a ChatCompletionAssistantMessageParam
        - response_output_message => Also produces/flushes a ChatCompletionAssistantMessageParam
        - tool calls get attached to the *current* assistant message, or create one if none.
        - tool outputs => ChatCompletionToolMessageParam
        """
        if isinstance(items, str):
            return [
                {
                    "role": "user",
                    "content": items,
                }
            ]

        result: list[ChatCompletionMessageParam] = []
        current_assistant_msg: dict[str, Any] | None = None

        def flush_assistant_message() -> None:
            nonlocal current_assistant_msg
            if current_assistant_msg is not None:
                if not current_assistant_msg.get("tool_calls"):
                    del current_assistant_msg["tool_calls"]
                result.append(cast(ChatCompletionMessageParam, current_assistant_msg))
                current_assistant_msg = None

        def ensure_assistant_message() -> dict[str, Any]:
            nonlocal current_assistant_msg
            if current_assistant_msg is None:
                current_assistant_msg = {"role": "assistant"}
                current_assistant_msg["tool_calls"] = []
            return current_assistant_msg

        for item in items:
            # 1) Check easy input message
            if easy_msg := cls.maybe_easy_input_message(item):
                role = easy_msg["role"]
                content = easy_msg["content"]

                if role == "user":
                    flush_assistant_message()
                    msg_user: ChatCompletionMessageParam = {
                        "role": "user",
                        "content": cls.extract_all_content(content),
                    }
                    result.append(msg_user)
                elif role == "system":
                    flush_assistant_message()
                    msg_system: ChatCompletionMessageParam = {
                        "role": "system",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_system)
                elif role == "developer":
                    flush_assistant_message()
                    msg_developer: ChatCompletionMessageParam = {
                        "role": "developer",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_developer)
                elif role == "assistant":
                    flush_assistant_message()
                    msg_assistant: ChatCompletionMessageParam = {
                        "role": "assistant",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_assistant)
                else:
                    raise UsageError(f"Unexpected role in easy_input_message: {role}")

            # 2) Check input message
            elif in_msg := cls.maybe_input_message(item):
                role = in_msg["role"]
                content = in_msg["content"]
                flush_assistant_message()

                if role == "user":
                    msg_user = {
                        "role": "user",
                        "content": cls.extract_all_content(content),
                    }
                    result.append(msg_user)
                elif role == "system":
                    msg_system = {
                        "role": "system",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_system)
                elif role == "developer":
                    msg_developer = {
                        "role": "developer",
                        "content": cls.extract_text_content(content),
                    }
                    result.append(msg_developer)
                else:
                    raise UsageError(f"Unexpected role in input_message: {role}")

            # 3) response output message => assistant
            elif resp_msg := cls.maybe_response_output_message(item):
                flush_assistant_message()
                new_asst = {"role": "assistant"}
                contents = resp_msg["content"]

                text_segments = []
                for c in contents:
                    if c["type"] == "output_text":
                        text_segments.append(c["text"])
                    elif c["type"] == "refusal":
                        new_asst["refusal"] = c["refusal"]
                    elif c["type"] == "output_audio":
                        raise AgentError(
                            f"Only audio IDs are supported for chat completions, but got: {c}"
                        )
                    else:
                        raise UsageError(f"Unknown content type in ResponseOutputMessage: {c}")

                if text_segments:
                    combined = "\n".join(text_segments)
                    new_asst["content"] = combined

                new_asst["tool_calls"] = []
                current_assistant_msg = new_asst

            # 4) function/file-search calls => attach to assistant
            elif file_search := cls.maybe_file_search_call(item):
                asst = ensure_assistant_message()
                tool_calls = list(asst.get("tool_calls", []))
                new_tool_call = {
                    "id": file_search["id"],
                    "type": "function",
                    "function": {
                        "name": "file_search_call",
                        "arguments": json.dumps(
                            {
                                "queries": file_search.get("queries", []),
                                "status": file_search.get("status"),
                            },
                            cls=NotGivenEncoder
                        ),
                    },
                }
                tool_calls.append(new_tool_call)
                asst["tool_calls"] = tool_calls

            elif func_call := cls.maybe_function_tool_call(item):
                asst = ensure_assistant_message()
                tool_calls = list(asst.get("tool_calls", []))
                new_tool_call = {
                    "id": func_call["call_id"],
                    "type": "function",
                    "function": {
                        "name": func_call["name"],
                        "arguments": func_call["arguments"],
                    },
                }
                tool_calls.append(new_tool_call)
                asst["tool_calls"] = tool_calls
            # 5) function call output => tool message
            elif func_output := cls.maybe_function_tool_call_output(item):
                flush_assistant_message()
                msg = {
                    "role": "tool",
                    "tool_call_id": func_output["call_id"],
                    "content": func_output["output"],
                }
                result.append(cast(ChatCompletionMessageParam, msg))

            # 6) item reference => handle or raise
            elif item_ref := cls.maybe_item_reference(item):
                raise UsageError(
                    f"Encountered an item_reference, which is not supported: {item_ref}"
                )

            # 7) If we haven't recognized it => fail or ignore
            else:
                raise UsageError(f"Unhandled item type or structure: {item}")

        flush_assistant_message()
        return result


########################################################
#           Main Class: Tool Converter                #
########################################################

class ToolConverter:
    """Converts tool definitions to API format."""

    @classmethod
    def to_api_format(cls, tool: Tool) -> ChatCompletionToolParam:
        """Converts a tool to its API representation."""
        if isinstance(tool, FunctionTool):
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.params_json_schema,
                },
            }

        raise AgentError(
            f"Chat completions API does not support hosted tools. Received tool type: "
            f"{type(tool)}, tool: {tool}"
        )

    @classmethod
    def convert_handoff_tool(cls, handoff: Handoff[Any]) -> ChatCompletionToolParam:
        """Converts a handoff tool to API format."""
        return {
            "type": "function",
            "function": {
                "name": handoff.tool_name,
                "description": handoff.tool_description,
                "parameters": handoff.input_json_schema,
            },
        }
