"""
Shields Module - Input and Output Validation Framework

This module provides a robust validation framework for agent input and output processing.
Shields act as protective layers that ensure data integrity and type safety throughout
the agent execution pipeline.

Key Components:
    - InputShield: Validates and sanitizes agent input before execution
    - OutputShield: Validates and formats agent output after execution

Features:
    - Type checking and validation
    - Data sanitization and normalization
    - Error handling and reporting
    - Custom validation rules support
    - Integration with agent execution pipeline
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, overload

from typing_extensions import TypeVar

from ..util._exceptions import UsageError
from ..util._items import TResponseInputItem
from ..util._types import MaybeAwaitable, RunContextWrapper, TContext

if TYPE_CHECKING:
    from ..agents.agent import Agent


########################################################
#               Data classes                           #
########################################################


@dataclass(frozen=True)
class ShieldFunctionOutput:
    """Output from a shield function."""

    tripwire_triggered: bool
    output: Any | None = None


@dataclass(frozen=True)
class InputShieldResult:
    """Result from running an input shield."""

    shield: InputShield[Any]
    agent: Agent[Any]
    input: str | list[TResponseInputItem]
    output: ShieldFunctionOutput


@dataclass(frozen=True)
class OutputShieldResult:
    """Result from running an output shield."""

    shield: OutputShield[Any]
    agent: Agent[Any]
    agent_output: Any
    output: ShieldFunctionOutput


@dataclass(frozen=True)
class InputShield(Generic[TContext]):
    """Shield that validates agent input before execution."""

    shield_function: Callable[
        [RunContextWrapper[TContext], Agent[Any], str | list[TResponseInputItem]],
        MaybeAwaitable[ShieldFunctionOutput],
    ]

    name: str | None = None

    async def run(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[Any],
        input: str | list[TResponseInputItem],
    ) -> InputShieldResult:
        if not callable(self.shield_function):
            raise UsageError(f"Shield function must be callable, got {self.shield_function}")

        if output := self.shield_function(context, agent, input):
            if inspect.isawaitable(output):
                return InputShieldResult(
                    shield=self,
                    agent=agent,
                    input=input,
                    output=await output,
                )

        return InputShieldResult(
            shield=self,
            agent=agent,
            input=input,
            output=output,
        )


@dataclass(frozen=True)
class OutputShield(Generic[TContext]):
    """Shield that validates agent output after execution."""

    shield_function: Callable[
        [RunContextWrapper[TContext], Agent[Any], Any],
        MaybeAwaitable[ShieldFunctionOutput],
    ]
    name: str | None = None

    async def run(
        self, context: RunContextWrapper[TContext], agent: Agent[Any], agent_output: Any
    ) -> OutputShieldResult:
        if not callable(self.shield_function):
            raise UsageError(f"Shield function must be callable, got {self.shield_function}")

        if output := self.shield_function(context, agent, agent_output):
            if inspect.isawaitable(output):
                return OutputShieldResult(
                    shield=self,
                    agent=agent,
                    agent_output=agent_output,
                    output=await output,
                )

        return OutputShieldResult(
            shield=self,
            agent=agent,
            agent_output=agent_output,
            output=output,
        )


TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)
_InputShieldFuncSync = Callable[
    [
        RunContextWrapper[TContext_co],
        "Agent[Any]",
        str | list[TResponseInputItem],
    ],
    ShieldFunctionOutput,
]
_InputShieldFuncAsync = Callable[
    [
        RunContextWrapper[TContext_co],
        "Agent[Any]",
        str | list[TResponseInputItem],
    ],
    Awaitable[ShieldFunctionOutput],
]


@overload
def input_shield(
    func: _InputShieldFuncSync[TContext_co],
) -> InputShield[TContext_co]: ...


@overload
def input_shield(
    func: _InputShieldFuncAsync[TContext_co],
) -> InputShield[TContext_co]: ...


@overload
def input_shield(
    *,
    name: str | None = None,
) -> Callable[
    [_InputShieldFuncSync[TContext_co] | _InputShieldFuncAsync[TContext_co]],
    InputShield[TContext_co],
]: ...


def input_shield(
    func: _InputShieldFuncSync[TContext_co] | _InputShieldFuncAsync[TContext_co] | None = None,
    *,
    name: str | None = None,
) -> (
    InputShield[TContext_co]
    | Callable[
        [_InputShieldFuncSync[TContext_co] | _InputShieldFuncAsync[TContext_co]],
        InputShield[TContext_co],
    ]
):
    """Decorator for creating InputShields."""

    def decorator(
        f: _InputShieldFuncSync[TContext_co] | _InputShieldFuncAsync[TContext_co],
    ) -> InputShield[TContext_co]:
        return InputShield(shield_function=f, name=name)

    if func is not None:
        return decorator(func)

    return decorator


_OutputShieldFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Any],
    ShieldFunctionOutput,
]
_OutputShieldFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Any],
    Awaitable[ShieldFunctionOutput],
]


@overload
def output_shield(
    func: _OutputShieldFuncSync[TContext_co],
) -> OutputShield[TContext_co]: ...


@overload
def output_shield(
    func: _OutputShieldFuncAsync[TContext_co],
) -> OutputShield[TContext_co]: ...


@overload
def output_shield(
    *,
    name: str | None = None,
) -> Callable[
    [_OutputShieldFuncSync[TContext_co] | _OutputShieldFuncAsync[TContext_co]],
    OutputShield[TContext_co],
]: ...


def output_shield(
    func: _OutputShieldFuncSync[TContext_co] | _OutputShieldFuncAsync[TContext_co] | None = None,
    *,
    name: str | None = None,
) -> (
    OutputShield[TContext_co]
    | Callable[
        [_OutputShieldFuncSync[TContext_co] | _OutputShieldFuncAsync[TContext_co]],
        OutputShield[TContext_co],
    ]
):
    def decorator(
        f: _OutputShieldFuncSync[TContext_co] | _OutputShieldFuncAsync[TContext_co],
    ) -> OutputShield[TContext_co]:
        return OutputShield(shield_function=f, name=name)

    if func is not None:
        return decorator(func)

    return decorator
