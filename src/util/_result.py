from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

from typing_extensions import TypeVar

from ..agents.agent import Agent
from ..agents.output import AgentOutputSchema
from ..agents.run_impl import QueueCompleteSentinel
from ._constants import MAX_GUARDRAIL_QUEUE_SIZE, MAX_QUEUE_SIZE
from ._env import get_env_var
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
#               Constants                               #
########################################################

MAX_QUEUE_SIZE = get_env_var("MAX_QUEUE_SIZE", MAX_QUEUE_SIZE)
MAX_GUARDRAIL_QUEUE_SIZE = get_env_var("MAX_GUARDRAIL_QUEUE_SIZE", MAX_GUARDRAIL_QUEUE_SIZE)

########################################################
#               Public Types                           #
########################################################

T = TypeVar("T")

########################################################
#               Data Classes                          #
########################################################
@dataclass(frozen=True)
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


@dataclass(frozen=True)
class RunResult(RunResultBase):
    _last_agent: Agent[Any]

    @property
    def last_agent(self) -> Agent[Any]:
        """Last agent that was run."""
        return self._last_agent

    def __str__(self) -> str:
        return pretty_print_result(self)


@dataclass(frozen=True)
class RunResultStreaming(RunResultBase):
    """Streaming agent run result. Raises MaxTurnsError or GuardrailTripwireTriggered on failure."""

    current_agent: Agent[Any]
    current_turn: int
    max_turns: int
    final_output: Any
    _current_agent_output_schema: AgentOutputSchema | None = field(repr=False)
    is_complete: bool = False

    # Optimized queue initialization with max sizes
    _event_queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel] = field(
        default_factory=lambda: asyncio.Queue(maxsize=MAX_QUEUE_SIZE),
        repr=False
    )
    _input_guardrail_queue: asyncio.Queue[InputGuardrailResult] = field(
        default_factory=lambda: asyncio.Queue(maxsize=MAX_GUARDRAIL_QUEUE_SIZE),
        repr=False
    )

    # Async tasks with improved type hints
    _run_impl_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _input_guardrails_task: asyncio.Task[list[InputGuardrailResult]] | None = field(default=None, repr=False)
    _output_guardrails_task: asyncio.Task[list[OutputGuardrailResult]] | None = field(default=None, repr=False)
    _stored_exception: Exception | None = field(default=None, repr=False)
    _active_tasks: ClassVar[set[asyncio.Task[Any]]] = set()

    @property
    def last_agent(self) -> Agent[Any]:
        """Last agent (updates during run, final value only after completion)."""
        return self.current_agent

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Stream semantic events as they're generated. Raises MaxTurnsError or GuardrailTripwireTriggered on failure."""
        try:
            while True:
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
                    break

                yield item
                self._event_queue.task_done()
        finally:
            self._cleanup_tasks()
            if self._stored_exception:
                raise self._stored_exception

    def _check_errors(self) -> None:
        """Optimized error checking that only processes necessary checks."""
        if self.current_turn > self.max_turns:
            self._stored_exception = MaxTurnsError(f"Max turns ({self.max_turns}) exceeded")
            return

        if not self._input_guardrail_queue.empty():
            try:
                guardrail_result = self._input_guardrail_queue.get_nowait()
                if guardrail_result.output.tripwire_triggered:
                    self._stored_exception = InputGuardrailError(guardrail_result)
            except asyncio.QueueEmpty:
                pass

        for task in (self._run_impl_task, self._input_guardrails_task, self._output_guardrails_task):
            if task and task.done():
                exc = task.exception()
                if exc and isinstance(exc, Exception):
                    self._stored_exception = exc

    def _cleanup_tasks(self) -> None:
        """Efficiently cleanup all active tasks."""
        tasks_to_cancel = {
            task for task in (self._run_impl_task, self._input_guardrails_task, self._output_guardrails_task)
            if task and not task.done()
        }

        for task in tasks_to_cancel:
            task.cancel()
            self._active_tasks.discard(task)

    def __str__(self) -> str:
        return pretty_print_run_result_streaming(self)
