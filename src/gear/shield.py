"""
Shields Module - Input and Output Validation Framework

This module provides a robust validation framework for agent input and output
processing. Shields act as protective layers that ensure data integrity and
type safety throughout the agent execution pipeline.

Key Components:
    - InputShield: Validates and sanitizes agent input before execution
    - OutputShield: Validates and formats agent output after execution
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, overload

from ..util._constants import ERROR_MESSAGES
from ..util._exceptions import UsageError
from ..util._items import TResponseInputItem
from ..util._types import MaybeAwaitable, RunContextWrapper, T, TContext, TContext_co

if TYPE_CHECKING:
    from ..agents.agent import Agent


########################################################
#           Main Dataclasses for Shields
########################################################


@dataclass(frozen=True)
class ShieldResult:
    """Result from a shield function."""

    tripwire_triggered: bool
    result: Any | None = None


@dataclass(frozen=True)
class ShieldResultWrapper(Generic[T]):
    """Wrapper for shield results with additional context."""

    shield: Any
    agent: Agent[Any]
    data: T
    output: ShieldResult


class BaseShield(Generic[T, TContext]):
    """Base class for shields with common functionality."""

    def __init__(
        self,
        shield_function: Callable[
            [RunContextWrapper[TContext], Agent[Any], T],
            MaybeAwaitable[ShieldResult],
        ],
        name: str | None = None,
    ):
        self.shield_function = shield_function
        self.name = name

    async def _run_common(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[Any],
        data: T,
    ) -> ShieldResult:
        if not callable(self.shield_function):
            error_msg = ERROR_MESSAGES.SHIELD_FUNCTION_ERROR.message.format(
                error=self.shield_function
            )
            raise UsageError(error_msg)

        output = self.shield_function(context, agent, data)
        if inspect.isawaitable(output):
            output = await output

        return output


########################################################
#          Dataclasses for Input Shields
########################################################


@dataclass(frozen=True)
class InputShieldResult:
    """Result from an input shield function."""

    tripwire_triggered: bool
    shield: Any
    agent: Agent[Any]
    input: str | list[TResponseInputItem]
    output: ShieldResult
    result: Any | None = None


class InputShield(BaseShield[str | list[TResponseInputItem], TContext]):
    """Shield that validates agent input before execution."""

    async def run(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[Any],
        input: str | list[TResponseInputItem],
    ) -> InputShieldResult:
        result = await self._run_common(context, agent, input)
        return InputShieldResult(
            tripwire_triggered=result.tripwire_triggered,
            result=result.result,
            shield=self,
            agent=agent,
            input=input,
            output=result,
        )


########################################################
#          Dataclasses for Output Shields
########################################################


@dataclass(frozen=True)
class OutputShieldResult:
    """Result from an output shield function."""

    tripwire_triggered: bool
    shield: Any
    agent: Agent[Any]
    agent_output: Any
    output: ShieldResult
    result: Any | None = None


class OutputShield(BaseShield[Any, TContext]):
    """Shield that validates agent output after execution."""

    async def run(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[Any],
        agent_output: Any,
    ) -> OutputShieldResult:
        result = await self._run_common(context, agent, agent_output)
        return OutputShieldResult(
            tripwire_triggered=result.tripwire_triggered,
            result=result.result,
            shield=self,
            agent=agent,
            agent_output=agent_output,
            output=result,
        )


########################################################
#       Decorator factory for shield
########################################################


def create_shield_decorator(
    shield_class: type[BaseShield[TContext_co, Any]],
    sync_func_type: type,
    async_func_type: type,
):

    @overload
    def decorator(
        func: sync_func_type,
    ) -> shield_class: ...

    @overload
    def decorator(
        func: async_func_type,
    ) -> shield_class: ...

    @overload
    def decorator(
        *,
        name: str | None = None,
    ) -> Callable[
        [sync_func_type | async_func_type],
        shield_class,
    ]: ...

    def decorator(
        func: sync_func_type | async_func_type | None = None,
        *,
        name: str | None = None,
    ) -> (
        shield_class
        | Callable[
            [sync_func_type | async_func_type],
            shield_class,
        ]
    ):
        def create_shield(
            f: sync_func_type | async_func_type,
        ) -> shield_class:
            return shield_class(shield_function=f, name=name)

        if func is not None:
            return create_shield(func)
        return create_shield

    return decorator


########################################################
#        Type aliases for Input Shields
########################################################


_InputShieldFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", str | list[TResponseInputItem]],
    ShieldResult,
]
_InputShieldFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", str | list[TResponseInputItem]],
    Awaitable[ShieldResult],
]

input_shield = create_shield_decorator(
    InputShield,
    _InputShieldFuncSync,
    _InputShieldFuncAsync,
)


########################################################
#        Type aliases for Output Shields
########################################################

_OutputShieldFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Any],
    ShieldResult,
]
_OutputShieldFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Any],
    Awaitable[ShieldResult],
]

output_shield = create_shield_decorator(
    OutputShield,
    _OutputShieldFuncSync,
    _OutputShieldFuncAsync,
)
