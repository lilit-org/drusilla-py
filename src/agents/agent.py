"""
This module implements the Agent class, which provides a flexible framework for creating AI agents.

The Agent class allows you to create AI agents with:
- Configurable instructions and model settings
- Tools for performing specific actions
- Input and output shield for safety and validation
- Give orbs to other agents or handlers
- Custom tool use behavior and output processing
- Hooks for monitoring and modifying agent behavior

Agents can be used standalone or converted into tools for other agents, enabling
composition of complex AI behaviors from simpler components.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, cast

from ..gear.orbs import Orbs
from ..gear.shields import InputShield, OutputShield
from ..models.settings import ModelSettings
from ..util import _json
from ..util._items import ItemHelpers
from ..util._run_context import RunContextWrapper, TContext
from ..util._tool import FunctionToolResult, Tool, function_tool
from ..util._types import MaybeAwaitable

if TYPE_CHECKING:
    from ..models.interface import Model
    from ..util._lifecycle import AgentHooks
    from ..util._result import RunResult


########################################################
#                    Private Methods
########################################################


def _create_agent_tool(
    agent: Agent[Any],
    tool_name: str | None,
    tool_description: str | None,
    custom_output_extractor: Callable[[RunResult], Awaitable[str]] | None,
) -> Tool:

    @function_tool(
        name_override=tool_name or _json.transform_string_function_style(agent.name),
        description_override=tool_description or "",
    )
    async def run_agent(context: RunContextWrapper, input: str) -> str:
        from .run import Runner

        output = await Runner.run(
            starting_agent=agent,
            input=input,
            context=context.context,
        )
        if custom_output_extractor:
            return await custom_output_extractor(output)

        return ItemHelpers.text_message_outputs(output.new_items)

    return run_agent


########################################################
#            Data Classes for Tools
########################################################


@dataclass
class ToolsToFinalOutputResult:
    is_final_output: bool
    final_output: Any | None = None


ToolsToFinalOutputFunction: TypeAlias = Callable[
    [RunContextWrapper[TContext], list[FunctionToolResult]],
    MaybeAwaitable[ToolsToFinalOutputResult],
]


########################################################
#            Main Data Class for Agents
########################################################


@dataclass
class Agent(Generic[TContext]):
    """AI agent with tools, shields, and orbs."""

    name: str
    instructions: (
        str
        | Callable[
            [RunContextWrapper[TContext], Agent[TContext]],
            MaybeAwaitable[str],
        ]
        | None
    ) = None

    orbs_description: str | None = None
    orbs: list[Agent[Any] | Orbs[TContext]] = field(default_factory=list)
    model: str | Model = field(default="")
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    tools: list[Tool] = field(default_factory=list)
    input_shields: list[InputShield[TContext]] = field(default_factory=list)
    output_shields: list[OutputShield[TContext]] = field(default_factory=list)
    output_type: type[Any] | None = None
    hooks: AgentHooks[TContext] | None = None
    tool_use_behavior: (
        Literal["run_llm_again", "stop_on_first_tool"]
        | dict[Literal["stop_at_tool_names"], list[str]]
        | ToolsToFinalOutputFunction
    ) = "run_llm_again"

    def clone(self, **kwargs: Any) -> Agent[TContext]:
        """Create a copy of the agent with optional field updates."""
        defaults = {
            "orbs": [],
            "model": "",
            "model_settings": ModelSettings(),
            "tools": [],
            "input_shields": [],
            "output_shields": [],
            "output_type": None,
            "hooks": None,
            "tool_use_behavior": "run_llm_again",
        }
        defaults.update(kwargs)
        return dataclasses.replace(self, **defaults)

    def as_tool(
        self,
        tool_name: str | None,
        tool_description: str | None,
        custom_output_extractor: Callable[[RunResult], Awaitable[str]] | None = None,
    ) -> Tool:
        """Converts agent to tool for other agents."""
        return _create_agent_tool(self, tool_name, tool_description, custom_output_extractor)

    async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
        """Get the system prompt for the agent."""
        if self.instructions is None:
            return None
        if isinstance(self.instructions, str):
            return self.instructions
        if callable(self.instructions):
            if inspect.iscoroutinefunction(self.instructions):
                return await cast(Awaitable[str], self.instructions(run_context, self))
            return cast(str, self.instructions(run_context, self))
        raise TypeError(f"Invalid instructions type: {type(self.instructions)}")
