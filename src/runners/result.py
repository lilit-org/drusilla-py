"""
This module provides data classes for handling the results of agent runs,
including both synchronous and streaming results.

Key Components:
1. RunResultBase: Abstract base class defining the core structure for all run results, including:
   - Input data and responses
   - Generated items and final output
   - Shield results for input and output validation
   - Agent tracking and status information

2. RunResult: Concrete implementation for completed run results, inheriting from RunResultBase

3. RunResultStreaming: Specialized implementation for streaming results that includes:
   - Asynchronous event streaming capabilities
   - Turn tracking and completion status
   - Internal task management for shields and execution
   - Exception handling and cleanup

4. SwordsToFinalOutputResult: Result indicating whether a sword execution produced a final output
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..util.types import QueueCompleteSentinel

if TYPE_CHECKING:
    from ..agents.agent_v1 import Agent
    from ..gear.shield import InputShieldResult, OutputShieldResult
    from ..util.types import (
        InputItem,
        RunResult,
        RunResultBase,
        RunResultStreaming,
        SwordsToFinalOutputResult,
    )
    from .items import ModelResponse, RunItem
    from .stream_events import StreamEvent

from ..util.constants import MAX_SHIELD_QUEUE_SIZE, logger

########################################################
#               Data Classes for Results
########################################################


@dataclass(frozen=True)
class SwordsToFinalOutputResult:
    """Result indicating whether a sword execution produced a final output."""

    is_final_output: bool
    final_output: Any | None


@dataclass(frozen=True)
class RunResultBase(abc.ABC):
    input: str | list[InputItem]
    new_items: list[RunItem]
    raw_responses: list[ModelResponse]
    final_output: Any
    input_shield_results: list[InputShieldResult]
    output_shield_results: list[OutputShieldResult]

    @property
    @abc.abstractmethod
    def last_agent(self) -> Agent[Any]:
        """Last agent that was run."""

    def __str__(self) -> str:
        stream_status = "Complete" if getattr(self, "is_complete", False) else "In Progress"
        return (
            f"✅ {self.__class__.__name__}:\n"
            f"Agent: {self.last_agent.name}\n"
            f"Stats: {len(self.new_items)} items, {len(self.raw_responses)} responses\n"
            f"Stream: {stream_status}\n"
            f"Final Output: {self.final_output}"
        )


@dataclass(frozen=True)
class RunResult(RunResultBase):
    _last_agent: Agent[Any]
    input_shield_results: list[InputShieldResult]
    output_shield_results: list[OutputShieldResult]
    final_output: Any
    is_complete: bool = True

    @property
    def last_agent(self) -> Agent[Any]:
        return self._last_agent

    def __post_init__(self):
        """Initialize the final output field."""
        if isinstance(self.input, str):
            object.__setattr__(self, "input", self.input.strip())


@dataclass
class RunResultStreaming:
    # Core result fields
    input: str | list[InputItem]
    new_items: list[RunItem]
    raw_responses: list[ModelResponse]
    input_shield_results: list[InputShieldResult]
    output_shield_results: list[OutputShieldResult]
    current_agent: Agent[Any]
    final_output: Any = None

    # Streaming-specific fields
    is_complete: bool = False
    current_turn: int = 0
    max_turns: int = 0

    # Internal state
    _event_queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel] = field(
        default_factory=lambda: asyncio.Queue(maxsize=MAX_SHIELD_QUEUE_SIZE), repr=False
    )
    _input_shield_queue: asyncio.Queue[InputShieldResult] = field(
        default_factory=lambda: asyncio.Queue(maxsize=MAX_SHIELD_QUEUE_SIZE),
        repr=False,
    )
    _run_impl_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _input_shields_task: asyncio.Task[list[InputShieldResult]] | None = field(
        default=None, repr=False
    )
    _output_shields_task: asyncio.Task[list[OutputShieldResult]] | None = field(
        default=None, repr=False
    )
    _stored_exception: Exception | None = field(default=None, repr=False)

    @property
    def last_agent(self) -> Agent[Any]:
        return self.current_agent

    def __post_init__(self):
        """Initialize the final output field and strip input string if it's a string."""
        if isinstance(self.input, str):
            object.__setattr__(self, "input", self.input.strip())

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        try:
            while True:
                if self._stored_exception:
                    logger.debug(f"Breaking stream: {self._stored_exception}")
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
                    self.is_complete = True
                    break

                yield item
                self._event_queue.task_done()
        finally:
            self._cleanup_tasks()
            if self._stored_exception:
                raise self._stored_exception

    def _cleanup_tasks(self) -> None:
        """Cancel any pending tasks that haven't completed."""
        tasks = (self._run_impl_task, self._input_shields_task, self._output_shields_task)
        for task in filter(lambda t: t and not t.done(), tasks):
            task.cancel()

    def __str__(self) -> str:
        stream_status = "Complete" if self.is_complete else "In Progress"
        return (
            f"✅ {self.__class__.__name__}:\n"
            f"Agent: {self.current_agent.name}\n"
            f"Stats: {len(self.new_items)} items, {len(self.raw_responses)} responses\n"
            f"Stream: {stream_status}\n"
            f"Final Output: {self.final_output}"
        )
