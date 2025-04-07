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
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from ..util._constants import ERROR_MESSAGES
from ..util._exceptions import UsageError
from ..util._types import (
    InputItem,
    MaybeAwaitable,
    ResponseInputItemParam,
    RunContextWrapper,
    T,
    TContext,
    TContext_co,
    create_decorator_factory,
)

if TYPE_CHECKING:
    from ..agents.agent import Agent


########################################################
#           Main Dataclasses for Shields
########################################################


@dataclass(frozen=True)
class ShieldResult:
    """Result of a shield validation operation."""

    success: bool
    message: str | None = None
    data: Any | None = None
    tripwire_triggered: bool = False
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
            error_msg = ERROR_MESSAGES.SHIELD_ERROR.message.format(error=self.shield_function)
            raise UsageError(error_msg)

        output = self.shield_function(context, agent, data)
        if inspect.isawaitable(output):
            output = await output

        return output


def create_shield_decorator(
    shield_class: type[BaseShield[TContext_co, Any]],
    sync_func_type: type,
    async_func_type: type,
):
    return create_decorator_factory(
        shield_class,
        sync_func_type,
        async_func_type,
        constructor_params={
            "shield_function": None,
            "name": None,
        },
    )


########################################################
#          Dataclasses for Input Shields
########################################################


@dataclass(frozen=True)
class InputShieldResult:
    """Result from an input shield function."""

    tripwire_triggered: bool
    shield: Any
    agent: Agent[Any]
    input: str | list[InputItem]
    output: ShieldResult
    result: Any | None = None


class InputShield(BaseShield[str | list[InputItem], TContext]):
    """Shield that validates agent input before execution."""

    async def run(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[Any],
        input: str | list[InputItem],
    ) -> InputShieldResult:
        result = await self._run_common(context, agent, input)
        return InputShieldResult(
            tripwire_triggered=not result.success or result.tripwire_triggered,
            result=result.message if not result.success else result.data,
            shield=self,
            agent=agent,
            input=input,
            output=result,
        )


# typeclass for input shield
InputShieldFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", str | list[ResponseInputItemParam]],
    ShieldResult,
]
InputShieldFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", str | list[ResponseInputItemParam]],
    MaybeAwaitable[ShieldResult],
]

# decorator for input shield
input_shield = create_shield_decorator(
    InputShield,
    InputShieldFuncSync,
    InputShieldFuncAsync,
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
            tripwire_triggered=not result.success or result.tripwire_triggered,
            result=result.message if not result.success else result.data,
            shield=self,
            agent=agent,
            agent_output=agent_output,
            output=result,
        )


# typeclass for output shield
OutputShieldFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Any],
    ShieldResult,
]
OutputShieldFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Any],
    MaybeAwaitable[ShieldResult],
]

# decorator for output shield
output_shield = create_shield_decorator(
    OutputShield,
    OutputShieldFuncSync,
    OutputShieldFuncAsync,
)
