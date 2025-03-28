from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, cast, overload

from pydantic import TypeAdapter
from typing_extensions import TypeAlias, TypeVar

from ..util._exceptions import UsageError
from ..util._items import RunItem, TResponseInputItem
from ..util._json import transform_string_function_style, validate_json
from ..util._run_context import RunContextWrapper, TContext
from ..util._strict_schema import ensure_strict_json_schema

if TYPE_CHECKING:
    from ..agents.agent import Agent


########################################################
#               Type aliases
########################################################

TOrbInput = TypeVar("TOrbInput", default=Any)
OnOrbWithInput = Callable[[RunContextWrapper[Any], TOrbInput], Any]
OnOrbWithoutInput = Callable[[RunContextWrapper[Any]], Any]


########################################################
#               Data classes for Orbs
########################################################


@dataclass(frozen=True)
class OrbInputData:
    input_history: str | tuple[TResponseInputItem, ...]
    pre_orb_items: tuple[RunItem, ...]
    new_items: tuple[RunItem, ...]


OrbInputFilter: TypeAlias = Callable[[OrbInputData], OrbInputData]
"""Filters input data passed to next agent."""


@dataclass
class Orb(Generic[TContext]):
    """Represents delegation of a task from one agent to another."""

    tool_name: str
    tool_description: str
    input_json_schema: dict[str, Any]
    on_invoke_orb: Callable[[RunContextWrapper[Any], str], Awaitable[Agent[TContext]]]
    """Invokes orb with:
    1. Orb run context
    2. LLM arguments as JSON string (empty if no input)
    Returns an agent."""

    agent_name: str
    input_filter: OrbInputFilter | None = None
    """Filters inputs passed to next agent. By default, next agent sees full conversation history.
    Can be used to remove older inputs or specific tools.
    Note: In streaming mode, no items are streamed from this function."""

    @classmethod
    def default_tool_name(cls, agent: Agent[Any]) -> str:
        return transform_string_function_style(f"transfer_to_{agent.name}")

    @classmethod
    def default_tool_description(cls, agent: Agent[Any]) -> str:
        return (
            f"Orb to the {agent.name} agent to handle the request. "
            f"{agent.orb_description or ''}"
        )


########################################################
#               Public methods for Orbs
########################################################


@overload
def orb(
    agent: Agent[TContext],
    *,
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    input_filter: Callable[[OrbInputData], OrbInputData] | None = None,
) -> Orb[TContext]: ...


@overload
def orb(
    agent: Agent[TContext],
    *,
    on_orb: OnOrbWithInput[TOrbInput],
    input_type: type[TOrbInput],
    tool_description_override: str | None = None,
    tool_name_override: str | None = None,
    input_filter: Callable[[OrbInputData], OrbInputData] | None = None,
) -> Orb[TContext]: ...


@overload
def orb(
    agent: Agent[TContext],
    *,
    on_orb: OnOrbWithoutInput,
    tool_description_override: str | None = None,
    tool_name_override: str | None = None,
    input_filter: Callable[[OrbInputData], OrbInputData] | None = None,
) -> Orb[TContext]: ...


def orb(
    agent: Agent[TContext],
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    on_orb: OnOrbWithInput[TOrbInput] | OnOrbWithoutInput | None = None,
    input_type: type[TOrbInput] | None = None,
    input_filter: Callable[[OrbInputData], OrbInputData] | None = None,
) -> Orb[TContext]:
    """Creates an orb to another agent.

    Args:
        agent: Target agent or function returning an agent
        tool_name_override: Custom name for orb tool
        tool_description_override: Custom description for orb tool
        on_orb: Function called when orb is invoked
        input_type: Type for validating orb input (if on_orb takes input)
        input_filter: Function to filter inputs passed to next agent
    """
    if bool(on_orb) != bool(input_type):
        raise UsageError(
            "You must provide either both on_input and input_type, or neither"
        )

    type_adapter: TypeAdapter[Any] | None = None
    input_json_schema: dict[str, Any] = {}

    if input_type is not None:
        if not callable(on_orb):
            raise UsageError("on_orb must be callable")

        sig = inspect.signature(on_orb)
        if len(sig.parameters) != 2:
            raise UsageError("on_orb must take two arguments: context and input")

        type_adapter = TypeAdapter(input_type)
        input_json_schema = type_adapter.json_schema()
    elif on_orb is not None:
        sig = inspect.signature(on_orb)
        if len(sig.parameters) != 1:
            raise UsageError("on_orb must take one argument: context")

    async def _invoke_orb(
        ctx: RunContextWrapper[Any], input_json: str | None = None
    ) -> Agent[Any]:
        if (
            input_type is not None
            and type_adapter is not None
            and input_json is not None
        ):
            validated_input = validate_json(
                json_str=input_json,
                type_adapter=type_adapter,
                partial=False,
            )
            input_func = cast(OnOrbWithInput[TOrbInput], on_orb)
            if inspect.iscoroutinefunction(input_func):
                await input_func(ctx, validated_input)
            else:
                input_func(ctx, validated_input)
        elif on_orb is not None:
            no_input_func = cast(OnOrbWithoutInput, on_orb)
            if inspect.iscoroutinefunction(no_input_func):
                await no_input_func(ctx)
            else:
                no_input_func(ctx)

        return agent

    tool_name = tool_name_override or Orb.default_tool_name(agent)
    tool_description = tool_description_override or Orb.default_tool_description(agent)
    input_json_schema = ensure_strict_json_schema(input_json_schema)

    return Orb(
        tool_name=tool_name,
        tool_description=tool_description,
        input_json_schema=input_json_schema,
        on_invoke_orb=_invoke_orb,
        input_filter=input_filter,
        agent_name=agent.name,
    )
