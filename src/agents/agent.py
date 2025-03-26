"""
This module implements the Agent class, which provides a flexible framework for creating AI agents.

The Agent class allows you to create AI agents with:
- Configurable instructions and model settings
- Tools for performing specific actions
- Input and output guardrails for safety and validation
- Handoffs to other agents or handlers
- Custom tool use behavior and output processing
- Hooks for monitoring and modifying agent behavior

Agents can be used standalone or converted into tools for other agents, enabling
composition of complex AI behaviors from simpler components.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, cast

from typing_extensions import TypeAlias, TypedDict

from ..models.settings import ModelSettings
from ..util import _json
from ..util._guardrail import InputGuardrail, OutputGuardrail
from ..util._handoffs import Handoff
from ..util._items import ItemHelpers, RunItem
from ..util._run_context import RunContextWrapper, TContext
from ..util._tool import Tool, function_tool, FunctionToolResult
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
#                    Aux Data Classes
########################################################

@dataclass
class ToolsToFinalOutputResult:
    is_final_output: bool
    final_output: Any | None = None


ToolsToFinalOutputFunction: TypeAlias = Callable[
    [RunContextWrapper[TContext], list[FunctionToolResult]],
    MaybeAwaitable[ToolsToFinalOutputResult],
]


@dataclass
class StopAtTools(TypedDict):
    stop_at_tool_names: list[str]


########################################################
#                    Main Data Class
########################################################

@dataclass
class Agent(Generic[TContext]):
    """AI agent with tools, guardrails, and handoffs."""

    name: str
    instructions: (
        str
        | Callable[
            [RunContextWrapper[TContext], Agent[TContext]],
            MaybeAwaitable[str],
        ]
        | None
    ) = None

    handoff_description: str | None = None
    handoffs: list[Agent[Any] | Handoff[TContext]] = field(default_factory=list)
    model: str | Model | None = None
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    tools: list[Tool] = field(default_factory=list)
    input_guardrails: list[InputGuardrail[TContext]] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail[TContext]] = field(default_factory=list)
    output_type: type[Any] | None = None
    hooks: AgentHooks[TContext] | None = None
    tool_use_behavior: (
        Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools | ToolsToFinalOutputFunction
    ) = "run_llm_again"

    """
    Tool handling:
    - run_llm_again: Run tools, feed results to LLM
    - stop_on_first_tool: Use first tool result
    - tool names: Stop on listed tools
    - function: Custom processing
    """

    def clone(self, **kwargs: Any) -> Agent[TContext]:
        return dataclasses.replace(self, **kwargs)

    def as_tool(
        self,
        tool_name: str | None,
        tool_description: str | None,
        custom_output_extractor: Callable[[RunResult], Awaitable[str]] | None = None,
    ) -> Tool:
        """Converts agent to tool for other agents.
        Unlike handoffs:
        - Gets generated input, not history
        - Called as tool, not conversation"""
        return _create_agent_tool(self, tool_name, tool_description, custom_output_extractor)

    async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
        if isinstance(self.instructions, str):
            return self.instructions
        elif callable(self.instructions):
            if inspect.iscoroutinefunction(self.instructions):
                return await cast(Awaitable[str], self.instructions(run_context, self))
            return cast(str, self.instructions(run_context, self))
