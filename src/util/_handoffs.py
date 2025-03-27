from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, cast, overload

from pydantic import TypeAdapter
from typing_extensions import TypeAlias, TypeVar

from ..util._exceptions import ModelError, UsageError
from ..util._items import RunItem, TResponseInputItem
from ..util._json import transform_string_function_style, validate_json
from ..util._run_context import RunContextWrapper, TContext
from ..util._strict_schema import ensure_strict_json_schema

if TYPE_CHECKING:
    from ..agents.agent import Agent


########################################################
#               Type aliases                           #
########################################################

THandoffInput = TypeVar("THandoffInput", default=Any)
OnHandoffWithInput = Callable[[RunContextWrapper[Any], THandoffInput], Any]
OnHandoffWithoutInput = Callable[[RunContextWrapper[Any]], Any]


########################################################
#               Data classes                           #
########################################################


@dataclass(frozen=True)
class HandoffInputData:
    input_history: str | tuple[TResponseInputItem, ...]
    pre_handoff_items: tuple[RunItem, ...]
    new_items: tuple[RunItem, ...]


HandoffInputFilter: TypeAlias = Callable[[HandoffInputData], HandoffInputData]
"""Filters input data passed to next agent."""


@dataclass
class Handoff(Generic[TContext]):
    """Represents delegation of a task from one agent to another."""

    tool_name: str
    tool_description: str
    input_json_schema: dict[str, Any]
    on_invoke_handoff: Callable[
        [RunContextWrapper[Any], str], Awaitable[Agent[TContext]]
    ]
    """Invokes handoff with:
    1. Handoff run context
    2. LLM arguments as JSON string (empty if no input)
    Returns an agent."""

    agent_name: str
    input_filter: HandoffInputFilter | None = None
    """Filters inputs passed to next agent. By default, next agent sees full conversation history.
    Can be used to remove older inputs or specific tools.
    Note: In streaming mode, no items are streamed from this function."""

    def get_transfer_message(self, agent: Agent[Any]) -> str:
        """Get the transfer message for the handoff."""
        return f"{{'assistant': '{agent.name}'}}"

    @classmethod
    def default_tool_name(cls, agent: Agent[Any]) -> str:
        return transform_string_function_style(f"transfer_to_{agent.name}")

    @classmethod
    def default_tool_description(cls, agent: Agent[Any]) -> str:
        return (
            f"Handoff to the {agent.name} agent to handle the request. "
            f"{agent.handoff_description or ''}"
        )


@overload
def handoff(
    agent: Agent[TContext],
    *,
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    input_filter: Callable[[HandoffInputData], HandoffInputData] | None = None,
) -> Handoff[TContext]: ...


@overload
def handoff(
    agent: Agent[TContext],
    *,
    on_handoff: OnHandoffWithInput[THandoffInput],
    input_type: type[THandoffInput],
    tool_description_override: str | None = None,
    tool_name_override: str | None = None,
    input_filter: Callable[[HandoffInputData], HandoffInputData] | None = None,
) -> Handoff[TContext]: ...


@overload
def handoff(
    agent: Agent[TContext],
    *,
    on_handoff: OnHandoffWithoutInput,
    tool_description_override: str | None = None,
    tool_name_override: str | None = None,
    input_filter: Callable[[HandoffInputData], HandoffInputData] | None = None,
) -> Handoff[TContext]: ...


def handoff(
    agent: Agent[TContext],
    tool_name_override: str | None = None,
    tool_description_override: str | None = None,
    on_handoff: OnHandoffWithInput[THandoffInput] | OnHandoffWithoutInput | None = None,
    input_type: type[THandoffInput] | None = None,
    input_filter: Callable[[HandoffInputData], HandoffInputData] | None = None,
) -> Handoff[TContext]:
    """Creates a handoff to another agent.

    Args:
        agent: Target agent or function returning an agent
        tool_name_override: Custom name for handoff tool
        tool_description_override: Custom description for handoff tool
        on_handoff: Function called when handoff is invoked
        input_type: Type for validating handoff input (if on_handoff takes input)
        input_filter: Function to filter inputs passed to next agent
    """
    if bool(on_handoff) != bool(input_type):
        raise UsageError(
            "You must provide either both on_input and input_type, or neither"
        )

    type_adapter: TypeAdapter[Any] | None = None
    input_json_schema: dict[str, Any] = {}

    if input_type is not None:
        if not callable(on_handoff):
            raise UsageError("on_handoff must be callable")

        sig = inspect.signature(on_handoff)
        if len(sig.parameters) != 2:
            raise UsageError("on_handoff must take two arguments: context and input")

        type_adapter = TypeAdapter(input_type)
        input_json_schema = type_adapter.json_schema()
    elif on_handoff is not None:
        sig = inspect.signature(on_handoff)
        if len(sig.parameters) != 1:
            raise UsageError("on_handoff must take one argument: context")

    async def _invoke_handoff(
        ctx: RunContextWrapper[Any], input_json: str | None = None
    ) -> Agent[Any]:
        if input_type is not None and type_adapter is not None:
            if input_json is None:
                raise ModelError(
                    "Handoff function expected non-null input, but got None"
                )

            validated_input = validate_json(
                json_str=input_json,
                type_adapter=type_adapter,
                partial=False,
            )
            input_func = cast(OnHandoffWithInput[THandoffInput], on_handoff)
            if inspect.iscoroutinefunction(input_func):
                await input_func(ctx, validated_input)
            else:
                input_func(ctx, validated_input)
        elif on_handoff is not None:
            no_input_func = cast(OnHandoffWithoutInput, on_handoff)
            if inspect.iscoroutinefunction(no_input_func):
                await no_input_func(ctx)
            else:
                no_input_func(ctx)

        return agent

    tool_name = tool_name_override or Handoff.default_tool_name(agent)
    tool_description = tool_description_override or Handoff.default_tool_description(
        agent
    )
    input_json_schema = ensure_strict_json_schema(input_json_schema)

    return Handoff(
        tool_name=tool_name,
        tool_description=tool_description,
        input_json_schema=input_json_schema,
        on_invoke_handoff=_invoke_handoff,
        input_filter=input_filter,
        agent_name=agent.name,
    )
