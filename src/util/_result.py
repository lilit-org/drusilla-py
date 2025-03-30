from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..agents.agent import Agent
from ..agents.run_impl import QueueCompleteSentinel
from ..gear.shields import InputShieldResult, OutputShieldResult
from ._constants import MAX_SHIELD_QUEUE_SIZE
from ._env import get_env_var
from ._items import ModelResponse, RunItem, TResponseInputItem
from ._logger import logger
from ._stream_events import StreamEvent

########################################################
#               Constants
########################################################

MAX_SHIELD_QUEUE_SIZE = get_env_var("MAX_SHIELD_QUEUE_SIZE", MAX_SHIELD_QUEUE_SIZE)

########################################################
#               Data Classes for Results
########################################################


@dataclass(frozen=True)
class RunResultBase(abc.ABC):
    input: str | list[TResponseInputItem]
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
        tool_choice = getattr(self, "tool_choice", "N/A")
        return (
            f"✅ {self.__class__.__name__}:\n"
            f"Agent: {self.last_agent.name}\n"
            f"Stats: {len(self.new_items)} items, {len(self.raw_responses)} responses\n"
            f"Stream: {stream_status}\n"
            f"Tool Choice: {tool_choice}\n"
            f"Final Output: {self.final_output}"
        )


@dataclass(frozen=True)
class RunResult(RunResultBase):
    _last_agent: Agent[Any]

    @property
    def last_agent(self) -> Agent[Any]:
        return self._last_agent


@dataclass
class RunResultStreaming:
    # Core result fields
    input: str | list[TResponseInputItem]
    new_items: list[RunItem]
    raw_responses: list[ModelResponse]
    final_output: Any
    input_shield_results: list[InputShieldResult]
    output_shield_results: list[OutputShieldResult]

    # Streaming-specific fields
    current_agent: Agent[Any]
    is_complete: bool = False
    current_turn: int = 0
    max_turns: int = 0

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

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
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

    def _cleanup_tasks(self) -> None:
        for task in (
            self._run_impl_task,
            self._input_shields_task,
            self._output_shields_task,
        ):
            if task and not task.done():
                task.cancel()
