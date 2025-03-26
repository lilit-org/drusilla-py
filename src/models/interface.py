from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..agents.output import AgentOutputSchema
from ..util._handoffs import Handoff
from ..util._items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from ..util._tool import Tool
from .settings import ModelSettings

if TYPE_CHECKING:
    from ..util._handoffs import Handoff


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
        handoffs: list[Handoff],
    ) -> ModelResponse:
        """Get a complete model response.

        Args:
            system_instructions: System prompt/instructions for the model
            input: Model input items or raw string input
            model_settings: Configuration parameters for the model
            tools: List of available tools the model can use
            output_schema: Optional schema defining the expected output format
            handoffs: List of available handoffs for model interactions

        Returns:
            A ModelResponse containing the model's output and usage statistics
        """
        pass

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Stream model responses as they are generated.

        Args:
            system_instructions: System prompt/instructions for the model
            input: Model input items or raw string input
            model_settings: Configuration parameters for the model
            tools: List of available tools the model can use
            output_schema: Optional schema defining the expected output format
            handoffs: List of available handoffs for model interactions

        Returns:
            An async iterator yielding response events as they are generated
        """
        pass


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
        """Get a model instance by name.

        Args:
            model_name: Optional model identifier. If None, a default model may be used.

        Returns:
            A configured Model instance ready for use
        """
        pass
