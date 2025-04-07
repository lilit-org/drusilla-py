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

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias

from pydantic import TypeAdapter

from ..agents.agent import Agent
from ..util._items import RunItem
from ..util._print import transform_string_function_style, validate_json
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


OrbsInputFilter: TypeAlias = Callable[[OrbsInputData], OrbsInputData]
"""Filters input data passed to next agent."""


@dataclass
class Orbs(Generic[TContext]):
    """Represents delegation of a task from one agent to another."""

    on_invoke_orbs: Callable[[RunContextWrapper[Any], str], Awaitable[Agent[TContext]]]
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


########################################################
#     Create orbs decorator factory
########################################################


def create_orbs_decorator(
    orbs_class: type[Orbs[TContext]],
    sync_func_type: type,
    async_func_type: type,
):
    def pre_init_hook(f: Any, params: dict[str, Any]) -> dict[str, Any]:
        type_adapter: TypeAdapter[Any] | None = None
        input_json_schema: dict[str, Any] = {}

        if hasattr(f, "__annotations__") and "input_type" in f.__annotations__:
            input_type = f.__annotations__["input_type"]
            type_adapter = TypeAdapter(input_type)
            input_json_schema = type_adapter.json_schema()

        async def on_invoke(
            ctx: RunContextWrapper[Any], input_json: str | None = None
        ) -> Agent[Any]:
            if type_adapter is not None and input_json is not None:
                validated_input = validate_json(
                    json_str=input_json,
                    type_adapter=type_adapter,
                    partial=False,
                )
                if inspect.iscoroutinefunction(f):
                    await f(ctx, validated_input)
                else:
                    f(ctx, validated_input)
            else:
                if inspect.iscoroutinefunction(f):
                    await f(ctx)
                else:
                    f(ctx)

            return ctx.agent

        params["on_invoke_orbs"] = on_invoke
        params["input_json_schema"] = input_json_schema
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
orbs = create_orbs_decorator(Orbs, OrbsFuncSync, OrbsFuncAsync)

__all__ = ["Orbs", "OrbsInputFilter", "orbs"]
