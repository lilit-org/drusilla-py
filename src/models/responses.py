"""
Model Response Management

This module provides the ModelResponsesModel class, a comprehensive response handler for
model interactions. It implements both synchronous and streaming response patterns,
with support for structured output formats and efficient response processing.

Key features:
- Synchronous response handling with structured output
- Streaming response support for real-time processing
- Response format validation and transformation
- Integration with model output schemas
- Support for both chat completions and function calling responses
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, overload

from ..agents.output import AgentOutputSchema
from ..gear.orbs import Orbs
from ..gear.sword import Sword
from ..util._constants import HEADERS, UNSET, logger
from ..util._items import ItemHelpers, ModelResponse, TResponseInputItem
from ..util._types import (
    AsyncDeepSeek,
    AsyncStream,
    ChatCompletionSwordChoiceOptionParam,
    ChatCompletionSwordParam,
    Response,
    ResponseFormat,
    ResponseOutput,
    Usage,
)
from .interface import Model
from .settings import ModelSettings

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

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        swords: list[Sword],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
    ) -> ModelResponse:
        try:
            response = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                swords,
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
        swords: list[Sword],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
    ) -> AsyncIterator[ResponseOutput]:
        try:
            stream = await self._fetch_response(
                system_instructions,
                input,
                model_settings,
                swords,
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
        swords: list[Sword],
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
        swords: list[Sword],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
        stream: Literal[False],
    ) -> Response: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        swords: list[Sword],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
        stream: Literal[True] | Literal[False] = False,
    ) -> Response | AsyncStream[ResponseOutput]:
        list_input = ItemHelpers.input_to_new_input_list(input)
        parallel_sword_calls = bool(model_settings.parallel_sword_calls and swords)
        sword_choice = Converter.convert_sword_choice(model_settings.sword_choice)
        converted_swords = Converter.convert_swords(swords, orbs)
        response_format = Converter.get_response_format(output_schema)

        logger.debug(
            f"Calling LLM {self.model} with input:\n"
            f"{json.dumps(list_input, indent=2)}\n"
            f"Swords:\n{json.dumps(converted_swords.swords, indent=2)}\n"
            f"Stream: {stream}\n"
            f"Sword choice: {sword_choice}\n"
            f"Response format: {response_format}\n"
        )

        return await self._client.responses.create(
            instructions=system_instructions,
            model=self.model,
            input=list_input,
            include=converted_swords.includes,
            swords=converted_swords.swords,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
            truncation=model_settings.truncation,
            max_output_tokens=model_settings.max_tokens,
            sword_choice=sword_choice,
            parallel_sword_calls=parallel_sword_calls or UNSET,
            stream=stream,
            extra_headers=HEADERS,
            text=response_format,
        )


########################################################
#             Main Class: Converter                    #
########################################################


@dataclass
class ConvertedSwords:
    swords: list[ChatCompletionSwordParam]


class Converter:
    @staticmethod
    def convert_sword_choice(
        sword_choice: Literal["auto", "required", "none"] | str | None,
    ) -> ChatCompletionSwordChoiceOptionParam:
        if sword_choice is None:
            return UNSET
        if sword_choice in ("required", "auto", "none"):
            return sword_choice
        if sword_choice in ("file_search", "computer_use_preview"):
            return {"type": sword_choice}
        return {"type": "function", "name": sword_choice}

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
    def convert_swords(
        cls,
        swords: list[Sword],
        orbs: list[Orbs[Any]],
    ) -> ConvertedSwords:
        converted_swords: list[ChatCompletionSwordParam] = []

        for sword in swords:
            converted_sword = cls._convert_sword(sword)
            converted_swords.append(converted_sword)

        converted_swords.extend(cls._convert_orb_sword(orb) for orb in orbs)

        return ConvertedSwords(swords=converted_swords)

    @staticmethod
    def _convert_sword(
        sword: Sword,
    ) -> tuple[ChatCompletionSwordParam]:
        return sword.params_json_schema

    @staticmethod
    def _convert_orb_sword(orbs: Orbs) -> ChatCompletionSwordParam:
        return {
            "name": orbs.sword_name,
            "description": orbs.sword_description,
            "parameters": orbs.input_json_schema,
            "strict": orbs.strict_json_schema,
            "type": "function",
        }
