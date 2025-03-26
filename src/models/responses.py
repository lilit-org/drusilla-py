from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

from ..agents.output import AgentOutputSchema
from ..util._constants import NOT_GIVEN, _HEADERS, _USER_AGENT, IncludeLiteral
from ..util._exceptions import UsageError
from ..util._handoffs import Handoff
from ..util._items import ItemHelpers, ModelResponse, TResponseInputItem
from ..util._logger import logger
from ..util._tool import ComputerTool, FileSearchTool, FunctionTool, Tool, WebSearchTool
from ..util._types import (
    AsyncDeepSeek,
    AsyncStream,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    Response,
    ResponseFormat,
    ResponseOutput,
)
from ..util._usage import Usage
from ..util._version import __version__
from .interface import Model
from .settings import ModelSettings

if TYPE_CHECKING:
    from .settings import ModelSettings


########################################################
#               Constants                              #
########################################################

_USER_AGENT = f"Agents/Python {__version__}"
_HEADERS = {"User-Agent": _USER_AGENT}
# API response
IncludeLiteral = Literal[
    "file_search_call.results",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
]


########################################################
#           Main Class: Responses Model                #
########################################################

class ModelResponsesModel(Model):
    """Model implementation using Model Responses API."""

    def __init__(
        self,
        model: str,
        model_client: AsyncDeepSeek,
    ) -> None:
        self.model = model
        self._client = model_client

    def _non_null_or_not_given(self, value: Any) -> Any:
        return value or NOT_GIVEN

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
    ) -> ModelResponse:
        try:
            response = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                stream=False,
            )

            logger.debug(
                "\n 🧠  LLM resp for responses:\n"
                f"{json.dumps(list(response.output), indent=2)}\n"
            )

            usage = (
                Usage(
                    requests=1,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                if response.usage
                else Usage()
            )

        except Exception as e:
            request_id = getattr(e, 'request_id', None)
            logger.error(f"Error getting response: {e}. (request_id: {request_id})")
            raise

        return ModelResponse(
            output=response.output,
            usage=usage,
            referenceable_id=response.id,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff]
    ) -> AsyncIterator[ResponseOutput]:
        """Yields a partial message as it is generated, as well as the usage information."""
        try:
            stream = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                stream=True,
            )

            async for chunk in stream:
                yield chunk

        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            raise

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        stream: Literal[True],
    ) -> AsyncStream[ResponseOutput]: ...

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        stream: Literal[False],
    ) -> Response: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        stream: Literal[True] | Literal[False] = False,
    ) -> Response | AsyncStream[ResponseOutput]:
        list_input = ItemHelpers.input_to_new_input_list(input)

        parallel_tool_calls = (
            True if model_settings.parallel_tool_calls and tools and len(tools) > 0 else NOT_GIVEN
        )

        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        converted_tools = Converter.convert_tools(tools, handoffs)
        response_format = Converter.get_response_format(output_schema)

        logger.debug(
            f"Calling LLM {self.model} with input:\n"
            f"{json.dumps(list_input, indent=2)}\n"
            f"Tools:\n{json.dumps(converted_tools.tools, indent=2)}\n"
            f"Stream: {stream}\n"
            f"Tool choice: {tool_choice}\n"
            f"Response format: {response_format}\n"
            )

        return await self._client.responses.create(
            instructions=self._non_null_or_not_given(system_instructions),
            model=self.model,
            input=list_input,
            include=converted_tools.includes,
            tools=converted_tools.tools,
            temperature=self._non_null_or_not_given(model_settings.temperature),
            top_p=self._non_null_or_not_given(model_settings.top_p),
            truncation=self._non_null_or_not_given(model_settings.truncation),
            max_output_tokens=self._non_null_or_not_given(model_settings.max_tokens),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            extra_headers=_HEADERS,
            text=response_format,
        )

    def _get_client(self) -> AsyncDeepSeek:
        return self._client or AsyncDeepSeek()


########################################################
#               Data Classes                          #
########################################################

@dataclass
class ConvertedTools:
    tools: list[ChatCompletionToolParam]
    includes: list[IncludeLiteral]


########################################################
#             Main Class: Converter                    #
########################################################

class Converter:
    """Tool conversion utilities."""

    @classmethod
    def convert_tool_choice(
        cls, tool_choice: Literal["auto", "required", "none"] | str | None
    ) -> ChatCompletionToolChoiceOptionParam:
        if tool_choice is None:
            return NOT_GIVEN
        elif tool_choice == "required":
            return "required"
        elif tool_choice == "auto":
            return "auto"
        elif tool_choice == "none":
            return "none"
        elif tool_choice == "file_search":
            return {
                "type": "file_search",
            }
        elif tool_choice == "web_search_preview":
            return {
                "type": "web_search_preview",
            }
        elif tool_choice == "computer_use_preview":
            return {
                "type": "computer_use_preview",
            }
        else:
            return {
                "type": "function",
                "name": tool_choice,
            }

    @classmethod
    def get_response_format(
        cls, output_schema: AgentOutputSchema | None
    ) -> ResponseFormat | None:
        if output_schema is None or output_schema.is_plain_text():
            return NOT_GIVEN
        else:
            return {
                "format": {
                    "type": "json_schema",
                    "name": "final_output",
                    "schema": output_schema.json_schema(),
                    "strict": output_schema.strict_json_schema,
                }
            }

    @classmethod
    def convert_tools(
        cls,
        tools: list[Tool],
        handoffs: list[Handoff[Any]],
    ) -> ConvertedTools:
        converted_tools: list[ChatCompletionToolParam] = []
        includes: list[IncludeLiteral] = []

        computer_tools = [tool for tool in tools if isinstance(tool, ComputerTool)]
        if len(computer_tools) > 1:
            raise UsageError(f"You can only provide one computer tool. Got {len(computer_tools)}")

        for tool in tools:
            converted_tool, include = cls._convert_tool(tool)
            converted_tools.append(converted_tool)
            if include:
                includes.append(include)

        for handoff in handoffs:
            converted_tools.append(cls._convert_handoff_tool(handoff))

        return ConvertedTools(tools=converted_tools, includes=includes)

    @classmethod
    def _convert_tool(cls, tool: Tool) -> tuple[ChatCompletionToolParam, IncludeLiteral | None]:
        """Convert tool to API format"""
        converted_tool: ChatCompletionToolParam = {"type": "function"}
        includes: IncludeLiteral | None = None

        if isinstance(tool, FunctionTool):
            converted_tool = {
                "name": tool.name,
                "parameters": tool.params_json_schema,
                "strict": tool.strict_json_schema,
                "type": "function",
                "description": tool.description,
            }
            includes = None
            return converted_tool, includes
        elif isinstance(tool, WebSearchTool):
            ws = {
                "type": "web_search_preview",
                "user_location": tool.user_location,
                "search_context_size": tool.search_context_size,
            }
            converted_tool = ws
            includes = None
            return converted_tool, includes
        elif isinstance(tool, FileSearchTool):
            converted_tool = {
                "type": "file_search",
                "vector_store_ids": tool.vector_store_ids,
            }
            if tool.max_num_results:
                converted_tool["max_num_results"] = tool.max_num_results
            if tool.ranking_options:
                converted_tool["ranking_options"] = tool.ranking_options
            if tool.filters:
                converted_tool["filters"] = tool.filters

            includes = "file_search_call.results" if tool.include_search_results else None
            return converted_tool, includes
        elif isinstance(tool, ComputerTool):
            converted_tool = {
                "type": "computer_use_preview",
                "environment": tool.computer.environment,
                "display_width": tool.computer.dimensions[0],
                "display_height": tool.computer.dimensions[1],
            }
            includes = None
            return converted_tool, includes

    @classmethod
    def _convert_handoff_tool(cls, handoff: Handoff) -> ChatCompletionToolParam:
        return {
            "name": handoff.tool_name,
            "parameters": handoff.input_json_schema,
            "strict": handoff.strict_json_schema,
            "type": "function",
            "description": handoff.tool_description,
        }
