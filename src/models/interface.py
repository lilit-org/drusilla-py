from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

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

class Model(abc.ABC):
    """Base interface for LLM calls."""

    @abc.abstractmethod
    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
    ) -> ModelResponse:
        """Get model response.

        Args:
            system_instructions: System prompt
            input: Model input items
            model_settings: Model config
            tools: Available tools
            output_schema: Output format
            handoffs: Available handoffs

        Returns:
            Model response
        """
        pass

    @abc.abstractmethod
    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Stream model response.

        Args:
            system_instructions: System prompt
            input: Model input items
            model_settings: Model config
            tools: Available tools
            output_schema: Output format
            handoffs: Available handoffs

        Returns:
            Stream of response events
        """
        pass


########################################################
#               Main Class for model providers         #
########################################################

class ModelProvider(abc.ABC):
    """Interface for model lookup."""

    @abc.abstractmethod
    def get_model(self, model_name: str | None) -> Model:
        """Get model by name.

        Args:
            model_name: Model identifier

        Returns:
            Model instance
        """
        pass
