from __future__ import annotations

import abc
import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

from ..agents.agent import Agent
from ..agents.run_impl import QueueCompleteSentinel
from ..gear.shields import InputShieldResult, OutputShieldResult
from ._constants import MAX_GUARDRAIL_QUEUE_SIZE, MAX_QUEUE_SIZE
from ._env import get_env_var
from ._exceptions import GenericError
from ._items import ModelResponse, RunItem, TResponseInputItem
from ._logger import logger
from ._stream_events import StreamEvent

########################################################
#               Constants
########################################################

MAX_QUEUE_SIZE = get_env_var("MAX_QUEUE_SIZE", MAX_QUEUE_SIZE)
MAX_GUARDRAIL_QUEUE_SIZE = get_env_var(
    "MAX_GUARDRAIL_QUEUE_SIZE", MAX_GUARDRAIL_QUEUE_SIZE
)

########################################################
#               Public Types
########################################################

T = TypeVar("T")

########################################################
#               Data Classes for Results
########################################################


def _indent(text: str, indent_level: int) -> str:
    indent_string = "  " * indent_level
    return "\n".join(f"{indent_string}{line}" for line in text.splitlines())


def _format_final_output(raw_response: ModelResponse) -> str:
    try:
        output = raw_response.output[0]["text"]
        match = re.search(r"<think>(.*?)</think>(.*)", output, re.DOTALL)

        if match:
            reasoning = match.group(1).strip().encode().decode("unicode-escape")
            final_result = match.group(2).strip().encode().decode("unicode-escape")
        else:
            reasoning = ""
            final_result = output.strip("'").strip().encode().decode("unicode-escape")

        return f"\n\n✅ REASONING:\n\n{reasoning}\n\n✅ RESULT:\n\n{final_result}\n"

    except GenericError as e:
        print(f"Error formatting final output: {e}")
        return ""


def _format_agent_info(result: Any) -> str:
    info = ["\n👾 Agent Info:"]
    if hasattr(result, "is_complete"):
        info.extend(
            [
                f"      Name   → {result.current_agent.name}",
                f"      Turn   → {result.current_turn}/{result.max_turns}",
                f"      Status → {'✔️ Complete' if result.is_complete else '🟡 Running'}",
            ]
        )
    else:
        info.append(f"      Last Agent → {result.last_agent.name}")
    return "\n" + "\n".join(_indent(line, 1) for line in info)


def _format_stats(result: Any) -> str:
    stats = [
        "\n📊 Statistics:",
        f"      Items     → {len(result.new_items)}",
        f"      Responses → {len(result.raw_responses)}",
        f"      Input GR  → {len(result.input_guardrail_results)}",
        f"      Output GR → {len(result.output_guardrail_results)}",
    ]
    return "\n" + "\n".join(_indent(stat, 1) for stat in stats)


def _format_stream_info(stream: bool, tool_choice: Any) -> str:
    def format_obj(x: Any) -> str:
        if x is None or x is object():
            return "None"
        if isinstance(x, bool):
            return "✔️ Enabled" if x else "❌ Disabled"
        return str(x)

    info = [
        "\n🦾 Configuration:",
        f"      Streaming → {format_obj(stream)}",
        f"      Tool Mode → {format_obj(tool_choice)}",
    ]
    return "\n" + "\n".join(_indent(line, 1) for line in info)


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
        """Return pretty-printed string representation."""
        parts = [
            f"✅ {self.__class__.__name__}:",
            _format_agent_info(self),
            _format_stats(self),
            _format_stream_info(
                stream=hasattr(self, "is_complete"),
                tool_choice=getattr(self, "tool_choice", None),
            ),
            _format_final_output(self.raw_responses[0]),
        ]
        return "".join(parts)


@dataclass(frozen=True)
class RunResult(RunResultBase):
    _last_agent: Agent[Any]

    @property
    def last_agent(self) -> Agent[Any]:
        """Last agent that was run."""
        return self._last_agent


@dataclass(frozen=True)
class RunResultStreaming(RunResultBase):
    """Streaming agent run result. Raises MaxTurnsError or GuardrailTripwireTriggered on failure."""

    current_agent: Agent[Any]
    current_turn: int
    max_turns: int
    final_output: Any
    is_complete: bool = False

    # Optimized queue initialization with max sizes
    _event_queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel] = field(
        default_factory=lambda: asyncio.Queue(maxsize=MAX_QUEUE_SIZE), repr=False
    )
    _input_shield_queue: asyncio.Queue[InputShieldResult] = field(
        default_factory=lambda: asyncio.Queue(maxsize=MAX_GUARDRAIL_QUEUE_SIZE),
        repr=False,
    )

    # Async tasks with improved type hints
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
        """Last agent (updates during run, final value only after completion)."""
        return self.current_agent

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Stream semantic events as they're generated."""
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
        """Efficiently cleanup all active tasks."""
        for task in (
            self._run_impl_task,
            self._input_shields_task,
            self._output_shields_task,
        ):
            if task and not task.done():
                task.cancel()
