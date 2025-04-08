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

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic

from ..agents.agent_v1 import AgentV1 as Agent
from ..runners.items import RunItem
from ..util.constants import ERROR_MESSAGES
from ..util.exceptions import UsageError
from ..util.print import transform_string_function_style
from ..util.types import (
    InputItem,
    RunContextWrapper,
    T,
    TContext,
)

########################################################
#          Main dataclasses for Orbs
########################################################


@dataclass(frozen=True)
class OrbsInputData:
    """Data structure for input filtering during orbs operations."""

    input_history: str | tuple[InputItem, ...]
    pre_orbs_items: tuple[RunItem, ...]
    new_items: tuple[RunItem, ...]


# function that filters input data passed to the next agent
OrbsInputFilter = Callable[[OrbsInputData], OrbsInputData]


@dataclass
class Orbs(Generic[TContext]):
    """Represents delegation of a task from one agent to another."""

    on_invoke_orbs: Callable[[RunContextWrapper[TContext], str | None], Awaitable[Agent[TContext]]]
    name: str | None = None
    description: str | None = None
    input_json_schema: dict[str, Any] | None = None
    input_filter: OrbsInputFilter | None = None

    @classmethod
    def default_name(cls, agent: Agent[Any]) -> str:
        """Generate a default name for the orbs based on the agent name."""
        return transform_string_function_style(f"transfer_to_{agent.name}")

    @classmethod
    def default_description(cls, agent: Agent[Any]) -> str:
        """Generate a default description for the orbs based on the agent."""
        desc = agent.orbs_description or ""
        return f"Orbs to the {agent.name} agent to handle the request. {desc}"


########################################################
#     Create orbs decorators
########################################################


def create_orbs_decorator(
    agent: Agent[T],
    input_filter: OrbsInputFilter | None = None,
) -> Callable:
    """Create an orbs decorator for the given agent."""

    def decorator(f: Any) -> Orbs[T]:
        input_json_schema: dict[str, Any] | None = None
        if hasattr(f, "__annotations__"):
            input_type = f.__annotations__.get("input_data", None)
            if input_type is not None:
                f.input_type = input_type
                try:
                    input_json_schema = input_type.model_json_schema()
                except (AttributeError, TypeError) as e:
                    raise UsageError(ERROR_MESSAGES.ORBS_ERROR.message.format(error=str(e))) from e

        async def on_invoke(
            ctx: RunContextWrapper[T],
            input_json: str | None = None,
        ) -> Agent[T]:
            if hasattr(f, "input_type"):
                if input_json is None:
                    raise UsageError(
                        ERROR_MESSAGES.ORBS_ERROR.message.format(
                            error=(
                                f"{f.__name__}() missing 1 required "
                                "positional argument: 'input_data'"
                            )
                        )
                    )
                try:
                    input_data = f.input_type.model_validate_json(input_json)
                    await _invoke_function(f, ctx, input_data)
                except Exception as e:
                    raise UsageError(
                        ERROR_MESSAGES.ORBS_ERROR.message.format(
                            error=f"Invalid input JSON: {str(e)}"
                        )
                    ) from e
            else:
                await _invoke_function(f, ctx)
            return agent

        return Orbs(
            on_invoke_orbs=on_invoke,
            name=Orbs.default_name(agent),
            description=Orbs.default_description(agent),
            input_json_schema=input_json_schema,
            input_filter=input_filter,
        )

    return decorator


async def _invoke_function(
    func: Callable[..., Any],
    ctx: RunContextWrapper[Any],
    *args: Any,
) -> None:
    """Helper function to invoke a function, handling both sync and async cases."""
    if asyncio.iscoroutinefunction(func):
        await func(ctx, *args)
    else:
        func(ctx, *args)


# decorator for output orbs
orbs = create_orbs_decorator
