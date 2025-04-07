"""
This module defines the core interfaces for interacting with language models in the system.

The Model interface provides a standardized way to interact with different language models,
supporting both synchronous and streaming responses. It handles system instructions,
input processing, and integration with various tools and orbs for enhanced functionality.

The ModelProvider interface enables dynamic model instantiation and management,
allowing the system to work with different model implementations through a unified interface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..agents.output import AgentOutputSchema
from ..gear.sword import Sword
from ..util._items import ModelResponse
from ..util._types import InputItem, ResponseStreamEvent
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
        input: str | list[InputItem],
        model_settings: ModelSettings,
        swords: list[Sword],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orb],
    ) -> ModelResponse:
        """Get a complete model response."""

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[InputItem],
        model_settings: ModelSettings,
        swords: list[Sword],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orb],
    ) -> AsyncIterator[ResponseStreamEvent]:
        """Stream model responses as they are generated"""


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
