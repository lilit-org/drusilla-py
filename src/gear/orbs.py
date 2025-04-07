"""
Orbs Module - Agent Task Delegation Framework

This module implements a task delegation system that enables seamless collaboration
between agents through Orbs. Orbs serve as intelligent intermediaries that:

- Facilitate dynamic task transfer between agents
- Maintain context and state during delegation
- Enable flexible agent-to-agent communication
- Support complex multi-agent workflows
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from ..agents.agent import Agent
from ..util._exceptions import UsageError
from ..util._items import RunItem
from ..util._print import transform_string_function_style
from ..util._types import (
    InputItem,
    RunContextWrapper,
    TContext,
    TOrbsInput,
    create_decorator_factory,
)

if TYPE_CHECKING:
    from ..agents.agent import Agent


########################################################
#               Type aliases
########################################################

OnOrbsWithInput = Callable[[RunContextWrapper[Any], TOrbsInput], Any]
OnOrbsWithoutInput = Callable[[RunContextWrapper[Any]], Any]


########################################################
#          Main Classes
########################################################


@dataclass(frozen=True)
class OrbsInputData:
    input_history: str | tuple[InputItem, ...]
    pre_orbs_items: tuple[RunItem, ...]
    new_items: tuple[RunItem, ...]


OrbsInputFilter = Callable[[OrbsInputData], OrbsInputData]


@dataclass
class Orbs(Generic[TContext]):
    """Represents delegation of a task from one agent to another."""

    on_invoke_orbs: Callable[[RunContextWrapper[Any], str | None], Awaitable[Agent[TContext]]]
    name: str | None = None
    description: str | None = None
    input_json_schema: dict[str, Any] | None = None
    input_filter: OrbsInputFilter | None = None

    @classmethod
    def default_name(cls, agent: Agent[Any]) -> str:
        return transform_string_function_style(f"transfer_to_{agent.name}")

    @classmethod
    def default_description(cls, agent: Agent[Any]) -> str:
        return (
            f"Orbs to the {agent.name} agent to handle the request. "
            f"{agent.orbs_description or ''}"
        )

    def to_api_format(self) -> dict[str, Any]:
        """Convert the Orbs instance to a serializable format."""
        # Handle MagicMock case
        if hasattr(self, "_mock_return_value"):
            return {
                "name": getattr(self, "name", ""),
                "description": getattr(self, "description", ""),
                "parameters": getattr(self, "input_json_schema", {}),
            }

        return {
            "name": self.name if self.name is not None else "",
            "description": self.description or "",
            "parameters": self.input_json_schema or {},
        }


########################################################
#     Create orbs decorator factory
########################################################


def create_orbs_decorator(
    orbs_class: type[Orbs[TContext]],
    sync_func_type: type,
    async_func_type: type,
):
    def pre_init_hook(f: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Hook to set up orbs parameters before initialization."""
        input_json_schema = None
        if hasattr(f, "input_type"):
            try:
                input_json_schema = f.input_type.model_json_schema()
            except AttributeError as err:
                raise UsageError(
                    "input_type must be a Pydantic model with model_json_schema method"
                ) from err

        async def on_invoke(
            ctx: RunContextWrapper[Any], input_json: str | None = None
        ) -> Agent[Any]:
            if input_json is not None and hasattr(f, "input_type"):
                input_data = f.input_type.model_validate_json(input_json)
                await f(ctx, input_data)
            else:
                await f(ctx)

            return ctx.agent

        params["on_invoke_orbs"] = on_invoke
        params["input_json_schema"] = input_json_schema
        params["name"] = f.__name__  # Set name to function name by default
        params["description"] = None
        return params

    return create_decorator_factory(
        orbs_class,
        sync_func_type,
        async_func_type,
        constructor_params={
            "on_invoke_orbs": None,  # Will be replaced by pre_init_hook
            "name": None,
            "description": None,
            "input_json_schema": None,
            "input_filter": None,
        },
        pre_init_hook=pre_init_hook,
    )


# Type aliases for orbs functions
OrbsFuncSync = Callable[[RunContextWrapper[TContext]], Agent[TContext]]
OrbsFuncAsync = Callable[[RunContextWrapper[TContext]], Awaitable[Agent[TContext]]]


# Decorator for orbs
def orbs(agent: Agent[Any], input_filter: OrbsInputFilter | None = None) -> Callable:
    """Create an orbs decorator for the given agent."""

    def decorator(f: Any) -> Orbs[Any]:
        # Validate function signature
        if hasattr(f, "__code__"):
            arg_count = f.__code__.co_argcount
            if arg_count > 2:
                raise UsageError("on_orbs must take at most two arguments: context and input")

        # Validate input type if present
        input_json_schema = None
        if hasattr(f, "input_type"):
            try:
                input_json_schema = f.input_type.model_json_schema()
            except (AttributeError, TypeError) as err:
                raise UsageError(
                    "input_type must be a Pydantic model with model_json_schema method"
                ) from err

        async def on_invoke(
            ctx: RunContextWrapper[Any], input_json: str | None = None
        ) -> Agent[Any]:
            if input_json is not None and hasattr(f, "input_type"):
                input_data = f.input_type.model_validate_json(input_json)
                await f(ctx, input_data)
            else:
                await f(ctx)
            return agent

        # Create the orbs instance with proper attributes
        try:
            orb = Orbs(
                on_invoke_orbs=on_invoke,
                name=Orbs.default_name(agent),
                description=Orbs.default_description(agent),
                input_json_schema=input_json_schema,
                input_filter=input_filter,
            )
            return orb
        except (AttributeError, TypeError) as err:
            raise UsageError(
                "input_type must be a Pydantic model with model_json_schema method"
            ) from err

    return decorator


__all__ = ["Orbs", "OrbsInputFilter", "orbs"]
