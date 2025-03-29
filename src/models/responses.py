from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, overload

from ..agents.output import AgentOutputSchema
from ..gear.orbs import Orbs
from ..util._constants import HEADERS, UNSET, IncludeLiteral
from ..util._exceptions import UsageError
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
from .interface import Model
from .settings import ModelSettings

########################################################
#               Data Classes                          #
########################################################


@dataclass
class ConvertedTools:
    tools: list[ChatCompletionToolParam]
    includes: list[IncludeLiteral]


########################################################
#               Main Class: Responses Model            #
########################################################


class ModelResponsesModel(Model):
    def __init__(
        self,
        model: str,
        model_client: AsyncDeepSeek,
    ) -> None:
        self.model = model
        self._client = model_client

    def _non_null_or_not_given(self, value: Any) -> Any:
        return value or UNSET

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
    ) -> ModelResponse:
        try:
            response = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                orbs,
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

            return ModelResponse(
                output=response.output,
                usage=usage,
                referenceable_id=response.id,
            )

        except Exception as e:
            request_id = getattr(e, "request_id", None)
            logger.error(f"Error getting response: {e}. (request_id: {request_id})")
            raise

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
    ) -> AsyncIterator[ResponseOutput]:
        try:
            stream = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                orbs,
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
        orbs: list[Orbs],
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
        orbs: list[Orbs],
        stream: Literal[False],
    ) -> Response: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
        stream: Literal[True] | Literal[False] = False,
    ) -> Response | AsyncStream[ResponseOutput]:
        list_input = ItemHelpers.input_to_new_input_list(input)
        parallel_tool_calls = bool(model_settings.parallel_tool_calls and tools)
        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        converted_tools = Converter.convert_tools(tools, orbs)
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
            parallel_tool_calls=parallel_tool_calls or UNSET,
            stream=stream,
            extra_headers=HEADERS,
            text=response_format,
        )


########################################################
#             Main Class: Converter                    #
########################################################


class Converter:
    @staticmethod
    def convert_tool_choice(
        tool_choice: Literal["auto", "required", "none"] | str | None,
    ) -> ChatCompletionToolChoiceOptionParam:
        if tool_choice is None:
            return UNSET
        if tool_choice in ("required", "auto", "none"):
            return tool_choice
        if tool_choice in ("file_search", "web_search_preview", "computer_use_preview"):
            return {"type": tool_choice}
        return {"type": "function", "name": tool_choice}

    @staticmethod
    def get_response_format(
        output_schema: AgentOutputSchema | None,
    ) -> ResponseFormat | None:
        if not output_schema or output_schema.is_plain_text():
            return UNSET
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
        orbs: list[Orbs[Any]],
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

        converted_tools.extend(cls._convert_orb_tool(orb) for orb in orbs)

        return ConvertedTools(tools=converted_tools, includes=includes)

    @staticmethod
    def _convert_tool(
        tool: Tool,
    ) -> tuple[ChatCompletionToolParam, IncludeLiteral | None]:
        if isinstance(tool, FunctionTool):
            return {
                "name": tool.name,
                "parameters": tool.params_json_schema,
                "strict": tool.strict_json_schema,
                "type": "function",
                "description": tool.description,
            }, None
        if isinstance(tool, WebSearchTool):
            return {
                "type": "web_search_preview",
                "user_location": tool.user_location,
                "search_context_size": tool.search_context_size,
            }, None
        if isinstance(tool, FileSearchTool):
            converted_tool: ChatCompletionToolParam = {
                "type": "file_search",
                "vector_store_ids": tool.vector_store_ids,
            }
            if tool.max_num_results:
                converted_tool["max_num_results"] = tool.max_num_results
            if tool.ranking_options:
                converted_tool["ranking_options"] = tool.ranking_options
            if tool.filters:
                converted_tool["filters"] = tool.filters
            return converted_tool, (
                "file_search_call.results" if tool.include_search_results else None
            )
        if isinstance(tool, ComputerTool):
            return {
                "type": "computer_use_preview",
                "environment": tool.computer.environment,
                "display_width": tool.computer.dimensions[0],
                "display_height": tool.computer.dimensions[1],
            }, None

    @staticmethod
    def _convert_orb_tool(orbs: Orbs) -> ChatCompletionToolParam:
        return {
            "name": orbs.tool_name,
            "parameters": orbs.input_json_schema,
            "strict": orbs.strict_json_schema,
            "type": "function",
            "description": orbs.tool_description,
        }
