from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ..agents.output import AgentOutputSchema
from ..gear.orbs import Orb
from ..util._constants import FAKE_RESPONSES_ID, HEADERS, UNSET
from ..util._exceptions import AgentError, UsageError
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
    ResponseEvent,
    ResponseFormat,
    ResponseFunctionToolCall,
    ResponseOutputText,
)
from ..util._usage import Usage
from .interface import Model
from .settings import ModelSettings

########################################################
#           Private Class: Streaming State
########################################################


@dataclass
class _StreamingState:
    """Maintains the current state of streaming responses."""

    text_content_index_and_output: tuple[int, ResponseOutputText] | None = None
    function_calls: dict[int, ResponseFunctionToolCall] = field(default_factory=dict)


########################################################
#           Main Class: Chat Completions Model
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
        return None if not value else value

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orb],
    ) -> ModelResponse:
        response = await self._fetch_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            orbs,
            stream=False,
        )

        logger.debug("\n✅ Successfully received model response...")
        response_obj = response[0] if isinstance(response, tuple) else response

        usage = (
            Usage(
                requests=1,
                input_tokens=response_obj["usage"]["prompt_tokens"],
                output_tokens=response_obj["usage"]["completion_tokens"],
                total_tokens=response_obj["usage"]["total_tokens"],
            )
            if response_obj.get("usage")
            else Usage()
        )

        items = _Converter.message_to_output_items(response_obj["choices"][0]["message"])

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
        orbs: list[Orb],
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Stream model responses as generated."""
        response, stream = await self._fetch_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            orbs,
            stream=True,
        )

        if not isinstance(stream, AsyncStream):
            raise TypeError(f"Expected AsyncStream, got {type(stream)}")

        state = _StreamingState()
        async for chunk in stream:
            try:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    if not state.text_content_index_and_output:
                        state.text_content_index_and_output = (
                            0,
                            ResponseOutputText(
                                text=delta["content"],
                                type="output_text",
                                annotations=[],
                            ),
                        )
                        yield ResponseEvent(
                            type="content_part.added",
                            content_index=state.text_content_index_and_output[0],
                            item_id=FAKE_RESPONSES_ID,
                            output_index=0,
                            part=state.text_content_index_and_output[1],
                        )
                    else:
                        state.text_content_index_and_output[1]["text"] += delta["content"]
                        yield ResponseEvent(
                            type="output_text.delta",
                            content_index=state.text_content_index_and_output[0],
                            item_id=FAKE_RESPONSES_ID,
                            output_index=0,
                            delta=delta["content"],
                        )
            except Exception as e:
                logger.warning(f"Error processing chunk: {e}")
                continue

        if state.text_content_index_and_output:
            yield ResponseEvent(
                type="content_part.done",
                content_index=state.text_content_index_and_output[0],
                item_id=FAKE_RESPONSES_ID,
                output_index=0,
                part=state.text_content_index_and_output[1],
            )

        yield ResponseEvent(type="completed", response=response)

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orb],
        stream: bool = False,
    ) -> ChatCompletion | tuple[Response, AsyncStream]:
        converted_messages = _Converter.items_to_messages(input)

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )

        parallel_tool_calls = True if model_settings.parallel_tool_calls and tools else UNSET
        tool_choice = _Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = _Converter.convert_response_format(output_schema)
        converted_tools = [ToolConverter.to_api_format(tool) for tool in tools] if tools else []
        converted_tools.extend(ToolConverter.convert_orb_tool(orb) for orb in orbs)

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
                tool_choice=(
                    cast(Literal["auto", "required", "none"], tool_choice)
                    if tool_choice != UNSET
                    else "auto"
                ),
                top_p=model_settings.top_p,
                temperature=model_settings.temperature,
                tools=[],
                parallel_tool_calls=parallel_tool_calls or False,
            )
            if not isinstance(ret, AsyncStream):
                raise TypeError(f"Expected AsyncStream, got {type(ret)}")
            return response, ret

        return ret

    def _get_client(self) -> AsyncDeepSeek:
        return self._client


########################################################
#           Private Class: Converter
########################################################


class _Converter:
    @classmethod
    def convert_tool_choice(
        cls, tool_choice: Literal["auto", "required", "none"] | str | None
    ) -> ChatCompletionToolChoiceOptionParam | Any:
        if tool_choice is None:
            return UNSET
        if tool_choice in ("auto", "required", "none"):
            return tool_choice
        return {
            "type": "function",
            "function": {"name": tool_choice},
        }

    @classmethod
    def convert_response_format(
        cls, final_output_schema: AgentOutputSchema | None
    ) -> ResponseFormat | Any:
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
    def message_to_output_items(
        cls, message: ChatCompletionMessage | dict[str, Any]
    ) -> list[TResponseOutputItem]:
        items: list[TResponseOutputItem] = []
        message_dict = message if isinstance(message, dict) else message.__dict__

        if message_dict.get("content"):
            items.append(
                ResponseOutputText(
                    text=message_dict["content"],
                    type="output_text",
                    annotations=[],
                )
            )
        if message_dict.get("tool_calls"):
            for tool_call in message_dict["tool_calls"]:
                items.append(
                    ResponseFunctionToolCall(
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                        type="function_call",
                        call_id=tool_call.get("id", ""),
                        referenceable_id=None,
                    )
                )

        return items

    @classmethod
    def maybe_easy_input_message(cls, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict) or item.keys() != {"content", "role"}:
            return None
        role = item.get("role")
        if role not in ("user", "assistant", "system", "developer") or "content" not in item:
            return None
        return cast(dict[str, Any], item)

    @classmethod
    def maybe_input_message(cls, item: Any) -> dict[str, Any] | None:
        if (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") in ("user", "system", "developer")
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
    def maybe_function_tool_call_output(cls, item: Any) -> dict[str, Any] | None:
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
                out.append({"type": "text", "text": c["text"]})
            elif c.get("type") == "input_image":
                if "image_url" not in c or not c["image_url"]:
                    raise AgentError(f"Only image URLs are supported for input_image {c}")
                out.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": c["image_url"],
                            "detail": c["detail"],
                        },
                    }
                )
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
        if isinstance(items, str):
            return [{"role": "user", "content": items}]

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
                current_assistant_msg = {"role": "assistant", "tool_calls": []}
            return current_assistant_msg

        for item in items:
            if easy_msg := cls.maybe_easy_input_message(item):
                role = easy_msg["role"]
                content = easy_msg["content"]

                flush_assistant_message()
                if role == "user":
                    result.append(
                        {
                            "role": "user",
                            "content": cls.extract_all_content(content),
                        }
                    )
                elif role == "system":
                    result.append(
                        {
                            "role": "system",
                            "content": cls.extract_text_content(content),
                        }
                    )
                elif role == "developer":
                    result.append(
                        {
                            "role": "developer",
                            "content": cls.extract_text_content(content),
                        }
                    )
                elif role == "assistant":
                    result.append(
                        {
                            "role": "assistant",
                            "content": cls.extract_text_content(content),
                        }
                    )
                else:
                    raise UsageError(f"Unexpected role in easy_input_message: {role}")

            elif in_msg := cls.maybe_input_message(item):
                role = in_msg["role"]
                content = in_msg["content"]
                flush_assistant_message()

                if role == "user":
                    result.append(
                        {
                            "role": "user",
                            "content": cls.extract_all_content(content),
                        }
                    )
                elif role == "system":
                    result.append(
                        {
                            "role": "system",
                            "content": cls.extract_text_content(content),
                        }
                    )
                elif role == "developer":
                    result.append(
                        {
                            "role": "developer",
                            "content": cls.extract_text_content(content),
                        }
                    )
                else:
                    raise UsageError(f"Unexpected role in input_message: {role}")

            elif resp_msg := cls.maybe_response_output_message(item):
                flush_assistant_message()
                new_asst = {"role": "assistant", "tool_calls": []}
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
                    new_asst["content"] = "\n".join(text_segments)

                current_assistant_msg = new_asst

            elif file_search := cls.maybe_file_search_call(item):
                asst = ensure_assistant_message()
                tool_calls = list(asst.get("tool_calls", []))
                tool_calls.append(
                    {
                        "id": file_search["id"],
                        "type": "function",
                        "function": {
                            "name": "file_search_call",
                            "arguments": json.dumps(
                                {
                                    "queries": file_search.get("queries", []),
                                    "status": file_search.get("status"),
                                },
                                default=lambda x: None if x is UNSET else x,
                            ),
                        },
                    }
                )
                asst["tool_calls"] = tool_calls

            elif func_call := cls.maybe_function_tool_call(item):
                asst = ensure_assistant_message()
                tool_calls = list(asst.get("tool_calls", []))
                tool_calls.append(
                    {
                        "id": func_call["call_id"],
                        "type": "function",
                        "function": {
                            "name": func_call["name"],
                            "arguments": func_call["arguments"],
                        },
                    }
                )
                asst["tool_calls"] = tool_calls

            elif func_output := cls.maybe_function_tool_call_output(item):
                flush_assistant_message()
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": func_output["call_id"],
                        "content": func_output["output"],
                    }
                )

            elif cls.maybe_item_reference(item):
                raise UsageError("Encountered an item_reference, which is not supported")

            else:
                raise UsageError(f"Unhandled item type or structure: {item}")

        flush_assistant_message()
        return result


########################################################
#           Main Class: Tool Converter
########################################################


class ToolConverter:
    @classmethod
    def to_api_format(cls, tool: Tool) -> ChatCompletionToolParam:
        if isinstance(tool, FunctionTool):
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.params_json_schema,
                },
            }
        raise AgentError(f"Received tool type: {type(tool)}, tool: {tool}")

    @classmethod
    def convert_orb_tool(cls, orb: Orb[Any]) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": orb.tool_name,
                "description": orb.tool_description,
                "parameters": orb.input_json_schema,
            },
        }
