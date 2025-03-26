from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, Union, overload

from typing_extensions import TypeVar

from ..util._exceptions import UsageError
from ..util._items import TResponseInputItem
from ..util._run_context import RunContextWrapper, TContext
from ..util._types import MaybeAwaitable

if TYPE_CHECKING:
    from ..agents.agent import Agent


########################################################
#               Data classes                           #
########################################################

@dataclass(frozen=True)
class GuardrailFunctionOutput:
    """Output from a guardrail function."""
    tripwire_triggered: bool
    output: Any | None = None


@dataclass(frozen=True)
class InputGuardrailResult:
    """Result from running an input guardrail."""
    guardrail: InputGuardrail[Any]
    agent: Agent[Any]
    input: str | list[TResponseInputItem]
    output: GuardrailFunctionOutput


@dataclass(frozen=True)
class OutputGuardrailResult:
    """Result from running an output guardrail."""
    guardrail: OutputGuardrail[Any]
    agent: Agent[Any]
    agent_output: Any
    output: GuardrailFunctionOutput


@dataclass(frozen=True)
class InputGuardrail(Generic[TContext]):
    """Guardrail that validates agent input before execution."""
    guardrail_function: Callable[
        [RunContextWrapper[TContext], Agent[Any], str | list[TResponseInputItem]],
        MaybeAwaitable[GuardrailFunctionOutput],
    ]

    name: str | None = None

    def get_name(self) -> str:
        if self.name:
            return self.name
        return self.guardrail_function.__name__

    async def run(
        self, context: RunContextWrapper[TContext], agent: Agent[Any], input: str | list[TResponseInputItem]
    ) -> InputGuardrailResult:
        if not callable(self.guardrail_function):
            raise UsageError(f"Guardrail function must be callable, got {self.guardrail_function}")

        output = self.guardrail_function(context, agent, input)
        if inspect.isawaitable(output):
            return InputGuardrailResult(
                guardrail=self,
                agent=agent,
                input=input,
                output=await output,
            )

        return InputGuardrailResult(
            guardrail=self,
            agent=agent,
            input=input,
            output=output,
        )


@dataclass(frozen=True)
class OutputGuardrail(Generic[TContext]):
    """Guardrail that validates agent output after execution."""

    guardrail_function: Callable[
        [RunContextWrapper[TContext], Agent[Any], Any],
        MaybeAwaitable[GuardrailFunctionOutput],
    ]
    name: str | None = None

    def get_name(self) -> str:
        if self.name:
            return self.name

        return self.guardrail_function.__name__

    async def run(
        self, context: RunContextWrapper[TContext], agent: Agent[Any], agent_output: Any
    ) -> OutputGuardrailResult:
        if not callable(self.guardrail_function):
            raise UsageError(f"Guardrail function must be callable, got {self.guardrail_function}")

        output = self.guardrail_function(context, agent, agent_output)
        if inspect.isawaitable(output):
            return OutputGuardrailResult(
                guardrail=self,
                agent=agent,
                agent_output=agent_output,
                output=await output,
            )

        return OutputGuardrailResult(
            guardrail=self,
            agent=agent,
            agent_output=agent_output,
            output=output,
        )


TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)
_InputGuardrailFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Union[str, list[TResponseInputItem]]],
    GuardrailFunctionOutput,
]
_InputGuardrailFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Union[str, list[TResponseInputItem]]],
    Awaitable[GuardrailFunctionOutput],
]


@overload
def input_guardrail(
    func: _InputGuardrailFuncSync[TContext_co],
) -> InputGuardrail[TContext_co]: ...


@overload
def input_guardrail(
    func: _InputGuardrailFuncAsync[TContext_co],
) -> InputGuardrail[TContext_co]: ...


@overload
def input_guardrail(
    *,
    name: str | None = None,
) -> Callable[
    [_InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co]],
    InputGuardrail[TContext_co],
]: ...


def input_guardrail(
    func: _InputGuardrailFuncSync[TContext_co]
    | _InputGuardrailFuncAsync[TContext_co]
    | None = None,
    *,
    name: str | None = None,
) -> (
    InputGuardrail[TContext_co]
    | Callable[
        [_InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co]],
        InputGuardrail[TContext_co],
    ]
):
    """Decorator for creating InputGuardrails. Can be used as @input_guardrail or @input_guardrail(name="name")."""

    def decorator(
        f: _InputGuardrailFuncSync[TContext_co] | _InputGuardrailFuncAsync[TContext_co],
    ) -> InputGuardrail[TContext_co]:
        return InputGuardrail(guardrail_function=f, name=name)

    if func is not None:
        return decorator(func)

    return decorator


_OutputGuardrailFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Any],
    GuardrailFunctionOutput,
]
_OutputGuardrailFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", Any],
    Awaitable[GuardrailFunctionOutput],
]


@overload
def output_guardrail(
    func: _OutputGuardrailFuncSync[TContext_co],
) -> OutputGuardrail[TContext_co]: ...


@overload
def output_guardrail(
    func: _OutputGuardrailFuncAsync[TContext_co],
) -> OutputGuardrail[TContext_co]: ...


@overload
def output_guardrail(
    *,
    name: str | None = None,
) -> Callable[
    [_OutputGuardrailFuncSync[TContext_co] | _OutputGuardrailFuncAsync[TContext_co]],
    OutputGuardrail[TContext_co],
]: ...


def output_guardrail(
    func: _OutputGuardrailFuncSync[TContext_co]
    | _OutputGuardrailFuncAsync[TContext_co]
    | None = None,
    *,
    name: str | None = None,
) -> (
    OutputGuardrail[TContext_co]
    | Callable[
        [_OutputGuardrailFuncSync[TContext_co] | _OutputGuardrailFuncAsync[TContext_co]],
        OutputGuardrail[TContext_co],
    ]
):
    """Decorator for creating OutputGuardrails. Can be used as @output_guardrail or @output_guardrail(name="name")."""

    def decorator(
        f: _OutputGuardrailFuncSync[TContext_co] | _OutputGuardrailFuncAsync[TContext_co],
    ) -> OutputGuardrail[TContext_co]:
        return OutputGuardrail(guardrail_function=f, name=name)

    if func is not None:
        return decorator(func)

    return decorator
