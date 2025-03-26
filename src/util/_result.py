from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, cast

from typing_extensions import TypeVar

from ..agents.agent import Agent
from ..agents.output import AgentOutputSchema
from ..agents.run_impl import QueueCompleteSentinel
from ._exceptions import InputGuardrailError, MaxTurnsError
from ._guardrail import InputGuardrailResult, OutputGuardrailResult
from ._items import ItemHelpers, ModelResponse, RunItem, TResponseInputItem
from ._logger import logger
from ._pretty_print import pretty_print_result, pretty_print_run_result_streaming
from ._stream_events import StreamEvent

if TYPE_CHECKING:
    from ..agents.agent import Agent
    from ..agents.run_impl import QueueCompleteSentinel


########################################################
#               Public Types                           #
########################################################

T = TypeVar("T")


########################################################
#               Data Classes                          #
########################################################
@dataclass
class RunResultBase(abc.ABC):
    input: str | list[TResponseInputItem]
    new_items: list[RunItem]
    raw_responses: list[ModelResponse]
    final_output: Any
    input_guardrail_results: list[InputGuardrailResult]
    output_guardrail_results: list[OutputGuardrailResult]

    @property
    @abc.abstractmethod
    def last_agent(self) -> Agent[Any]:
        """Last agent that was run."""

    def final_output_as(self, cls: type[T], raise_if_incorrect_type: bool = False) -> T:
        """Cast final output to type T. Raises TypeError if type mismatch and raise_if_incorrect_type=True."""
        if raise_if_incorrect_type and not isinstance(self.final_output, cls):
            raise TypeError(f"Final output is not of type {cls.__name__}")

        return cast(T, self.final_output)

    def to_input_list(self) -> list[TResponseInputItem]:
        """Create new input list by merging original input with new items."""
        original_items: list[TResponseInputItem] = ItemHelpers.input_to_new_input_list(self.input)
        new_items = [item.to_input_item() for item in self.new_items]

        return original_items + new_items

    def pretty_print(self) -> str:
        """Return pretty-printed string representation."""
        return str(self)


@dataclass
class RunResult(RunResultBase):
    _last_agent: Agent[Any]

    @property
    def last_agent(self) -> Agent[Any]:
        """Last agent that was run."""
        return self._last_agent

    def __str__(self) -> str:
        return pretty_print_result(self)


@dataclass
class RunResultStreaming(RunResultBase):
    """Streaming agent run result. Raises MaxTurnsError or GuardrailTripwireTriggered on failure."""

    current_agent: Agent[Any]
    current_turn: int
    max_turns: int
    final_output: Any
    _current_agent_output_schema: AgentOutputSchema | None = field(repr=False)
    is_complete: bool = False
    _event_queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel] = field(
        default_factory=asyncio.Queue, repr=False
    )
    _input_guardrail_queue: asyncio.Queue[InputGuardrailResult] = field(
        default_factory=asyncio.Queue, repr=False
    )

    # Async tasks
    _run_impl_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _input_guardrails_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _output_guardrails_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _stored_exception: Exception | None = field(default=None, repr=False)

    @property
    def last_agent(self) -> Agent[Any]:
        """Last agent (updates during run, final value only after completion)."""
        return self.current_agent

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Stream semantic events as they're generated. Raises MaxTurnsError or GuardrailTripwireTriggered on failure."""
        while True:
            self._check_errors()
            if self._stored_exception:
                logger.debug("Breaking due to stored exception")
                self.is_complete = True
                break
            if self.is_complete and self._event_queue.empty():
                break
            try:
                item = await self._event_queue.get()
            except asyncio.CancelledError:
                break
            if isinstance(item, QueueCompleteSentinel):
                self._event_queue.task_done()
                self._check_errors()
                break

            yield item
            self._event_queue.task_done()

        self._cleanup_tasks()

        if self._stored_exception:
            raise self._stored_exception

    def _check_errors(self):
        if self.current_turn > self.max_turns:
            self._stored_exception = MaxTurnsError(f"Max turns ({self.max_turns}) exceeded")

        while not self._input_guardrail_queue.empty():
            guardrail_result = self._input_guardrail_queue.get_nowait()
            if guardrail_result.output.tripwire_triggered:
                self._stored_exception = InputGuardrailError(guardrail_result)

        if self._run_impl_task and self._run_impl_task.done():
            exc = self._run_impl_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

        if self._input_guardrails_task and self._input_guardrails_task.done():
            exc = self._input_guardrails_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

        if self._output_guardrails_task and self._output_guardrails_task.done():
            exc = self._output_guardrails_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

    def _cleanup_tasks(self):
        if self._run_impl_task and not self._run_impl_task.done():
            self._run_impl_task.cancel()

        if self._input_guardrails_task and not self._input_guardrails_task.done():
            self._input_guardrails_task.cancel()

        if self._output_guardrails_task and not self._output_guardrails_task.done():
            self._output_guardrails_task.cancel()

    def __str__(self) -> str:
        return pretty_print_run_result_streaming(self)
