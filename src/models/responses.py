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
from typing import Literal, overload

from ..agents.agent_v1 import AgentV1OutputSchema as AgentOutputSchema
from ..gear.orbs import Orbs
from ..gear.sword import Sword
from ..runners.items import ItemHelpers, ModelResponse
from ..util.constants import HEADERS, UNSET, err, logger
from ..util.exceptions import ModelError
from ..util.types import (
    AsyncDeepSeek,
    AsyncStream,
    ChatCompletionSwordParam,
    InputItem,
    Response,
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
        input: str | list[InputItem],
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
            raise ModelError(err.MODEL_ERROR.format(error=str(e))) from e

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[InputItem],
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
            raise ModelError(err.MODEL_ERROR.format(error=str(e))) from e

    @overload
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[InputItem],
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
        input: str | list[InputItem],
        model_settings: ModelSettings,
        swords: list[Sword],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
        stream: Literal[False],
    ) -> Response: ...

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[InputItem],
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
            f"Swords:\n{json.dumps(converted_swords, indent=2)}\n"
            f"Stream: {stream}\n"
            f"Sword choice: {sword_choice}\n"
            f"Response format: {response_format}\n"
        )

        return await self._client.responses.create(
            instructions=system_instructions,
            model=self.model,
            input=list_input,
            include=converted_swords,
            swords=converted_swords,
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
    @classmethod
    def convert_sword_choice(cls, choice: str | None) -> str:
        """Convert sword choice to API format."""
        if choice is None:
            return "auto"
        return choice

    @classmethod
    def get_response_format(cls, output_schema: AgentOutputSchema | None) -> dict[str, str]:
        """Convert output schema to response format."""
        if output_schema is None:
            return {"type": "text"}
        return {"type": "json_object"}

    @classmethod
    def convert_swords(
        cls, swords: list[Sword], orbs: list[Orbs]
    ) -> list[ChatCompletionSwordParam]:
        """Convert swords and orbs to API format."""
        converted_swords = []
        for sword in swords:
            converted_swords.append(
                {
                    "name": sword.name,
                    "description": sword.description,
                    "parameters": sword.params_json_schema,
                }
            )
        for orb in orbs:
            converted_swords.append(
                {
                    "name": orb.sword_name,
                    "description": orb.sword_description,
                    "parameters": orb.params_json_schema,
                }
            )
        return converted_swords

    async def process_input(
        self,
        input: str | list[InputItem],
        system_instructions: str | None,
        model_settings: ModelSettings,
        swords: list[Sword],
    ) -> None:
        if isinstance(input, str):
            await self.process_input_string(input, system_instructions, model_settings, swords)
        else:
            await self.process_input_list(input, system_instructions, model_settings, swords)

    async def process_input_list(
        self,
        input: str | list[InputItem],
        system_instructions: str | None,
        model_settings: ModelSettings,
        swords: list[Sword],
    ) -> None:
        for item in input:
            await self.process_input_items(item, system_instructions, model_settings, swords)

    async def process_input_string(
        self,
        input: str | list[InputItem],
        system_instructions: str | None,
        model_settings: ModelSettings,
        swords: list[Sword],
    ) -> None:
        await self.process_input_items(input, system_instructions, model_settings, swords)

    async def process_input_items(
        self,
        input: str | list[InputItem],
        system_instructions: str | None,
        model_settings: ModelSettings,
        swords: list[Sword],
    ) -> None:
        if isinstance(input, str):
            await self.process_input_string(input, system_instructions, model_settings, swords)
        else:
            await self.process_input_items_list(input, system_instructions, model_settings, swords)

    async def process_input_items_list(
        self,
        input: str | list[InputItem],
        system_instructions: str | None,
        model_settings: ModelSettings,
        swords: list[Sword],
    ) -> None:
        for item in input:
            await self.process_input_items(item, system_instructions, model_settings, swords)
