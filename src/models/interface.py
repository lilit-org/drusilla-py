from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..agents.output import AgentOutputSchema
from ..util._items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from ..util._tool import Tool
from .settings import ModelSettings

if TYPE_CHECKING:
    from ..gear.orbs import Orb


########################################################
#               Main Class for models                  #
########################################################


@runtime_checkable
class Model(Protocol):
    """Base interface for LLM calls.

    This interface defines the contract for model implementations that can generate
    responses either synchronously or asynchronously via streaming.
    """

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Orb],
    ) -> ModelResponse:
        """Get a complete model response.

        Args:
            system_instructions: System prompt/instructions for the model. Can be None.
            input: Either a raw string input or a list of structured input items.
            model_settings: Configuration parameters for the model.
            tools: List of available tools the model can use for function calling.
            output_schema: Optional schema defining the expected output format and structure.
            handoffs: List of available handoffs for model interactions with other agents.

        Returns:
            A ModelResponse containing the model's output and usage statistics.
        """

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Orb],
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Stream model responses as they are generated.

        Args:
            system_instructions: System prompt/instructions for the model. Can be None.
            input: Either a raw string input or a list of structured input items.
            model_settings: Configuration parameters for the model.
            tools: List of available tools the model can use for function calling.
            output_schema: Optional schema defining the expected output format and structure.
            handoffs: List of available handoffs for model interactions with other agents.

        Returns:
            An async iterator yielding response events as they are generated. Each event
            represents a partial response chunk or completion status.
        """


########################################################
#               Main Class for model providers         #
########################################################


@runtime_checkable
class ModelProvider(Protocol):
    """Interface for model lookup and instantiation.

    This interface defines the contract for providers that can create and return
    Model instances based on a model name or identifier.
    """

    def get_model(self, model_name: str | None) -> Model:
        pass
