"""
Orbs Module - Agent Task Delegation Framework

This module implements a sophisticated task delegation system that enables seamless collaboration
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
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, overload

from pydantic import TypeAdapter
from typing_extensions import TypeVar

from ..util._exceptions import UsageError
from ..util._items import RunItem
from ..util._print import transform_string_function_style, validate_json
from ..util._strict_schema import ensure_strict_json_schema
from ..util._types import InputItem, RunContextWrapper, TContext

if TYPE_CHECKING:
    from ..agents.agent import Agent


########################################################
#               Type aliases
########################################################

TOrbsInput = TypeVar("TOrbsInput", default=Any)
OnOrbsWithInput = Callable[[RunContextWrapper[Any], TOrbsInput], Any]
OnOrbsWithoutInput = Callable[[RunContextWrapper[Any]], Any]


########################################################
#               Data classes for Orbs
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

    sword_name: str
    sword_description: str
    input_json_schema: dict[str, Any]
    on_invoke_orbs: Callable[[RunContextWrapper[Any], str], Awaitable[Agent[TContext]]]
    agent_name: str
    input_filter: OrbsInputFilter | None = None

    @classmethod
    def default_sword_name(cls, agent: Agent[Any]) -> str:
        return transform_string_function_style(f"transfer_to_{agent.name}")

    @classmethod
    def default_sword_description(cls, agent: Agent[Any]) -> str:
        return (
            f"Orbs to the {agent.name} agent to handle the request. "
            f"{agent.orbs_description or ''}"
        )


########################################################
#               Public methods for Orbs
########################################################


@overload
def orbs(
    agent: Agent[TContext],
    *,
    sword_name_override: str | None = None,
    sword_description_override: str | None = None,
    input_filter: Callable[[OrbsInputData], OrbsInputData] | None = None,
) -> Orbs[TContext]: ...


@overload
def orbs(
    agent: Agent[TContext],
    *,
    on_orbs: OnOrbsWithInput[TOrbsInput],
    input_type: type[TOrbsInput],
    sword_description_override: str | None = None,
    sword_name_override: str | None = None,
    input_filter: Callable[[OrbsInputData], OrbsInputData] | None = None,
) -> Orbs[TContext]: ...


@overload
def orbs(
    agent: Agent[TContext],
    *,
    on_orbs: OnOrbsWithoutInput,
    sword_description_override: str | None = None,
    sword_name_override: str | None = None,
    input_filter: Callable[[OrbsInputData], OrbsInputData] | None = None,
) -> Orbs[TContext]: ...


def orbs(
    agent: Agent[TContext],
    sword_name_override: str | None = None,
    sword_description_override: str | None = None,
    on_orbs: OnOrbsWithInput[TOrbsInput] | OnOrbsWithoutInput | None = None,
    input_type: type[TOrbsInput] | None = None,
    input_filter: Callable[[OrbsInputData], OrbsInputData] | None = None,
) -> Orbs[TContext]:
    if bool(on_orbs) != bool(input_type):
        raise UsageError("You must provide either both on_input and input_type, or neither")

    type_adapter: TypeAdapter[Any] | None = None
    input_json_schema: dict[str, Any] = {}

    if input_type is not None:
        if not callable(on_orbs):
            raise UsageError("on_orb must be callable")

        sig = inspect.signature(on_orbs)
        if len(sig.parameters) != 2:
            raise UsageError("on_orbs must take two arguments: context and input")

        type_adapter = TypeAdapter(input_type)
        input_json_schema = type_adapter.json_schema()
    elif on_orbs is not None:
        sig = inspect.signature(on_orbs)
        if len(sig.parameters) != 1:
            raise UsageError("on_orbs must take one argument: context")

    async def _invoke_orbs(
        ctx: RunContextWrapper[Any], input_json: str | None = None
    ) -> Agent[Any]:
        if input_type is not None and type_adapter is not None and input_json is not None:
            validated_input = validate_json(
                json_str=input_json,
                type_adapter=type_adapter,
                partial=False,
            )
            if inspect.iscoroutinefunction(on_orbs):
                await on_orbs(ctx, validated_input)
            else:
                on_orbs(ctx, validated_input)
        elif on_orbs is not None:
            if inspect.iscoroutinefunction(on_orbs):
                await on_orbs(ctx)
            else:
                on_orbs(ctx)

        return agent

    sword_name = sword_name_override or Orbs.default_sword_name(agent)
    sword_description = sword_description_override or Orbs.default_sword_description(agent)
    input_json_schema = ensure_strict_json_schema(input_json_schema)

    return Orbs(
        sword_name=sword_name,
        sword_description=sword_description,
        input_json_schema=input_json_schema,
        on_invoke_orbs=_invoke_orbs,
        input_filter=input_filter,
        agent_name=agent.name,
    )
