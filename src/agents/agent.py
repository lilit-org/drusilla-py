"""
The Agent module provides a powerful framework for creating and managing AI agents
with advanced capabilities.

This module implements the core Agent class that enables the creation of
sophisticated AI agents with:
- Function calling capabilities through swords
- Input/output validation through shields
- Context management through orbs
- Custom behavior modification through charms
- Flexible model integration and configuration
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, cast

if TYPE_CHECKING:
    from ..gear.charm import AgentCharms
    from ..gear.orbs import Orbs
    from ..models.interface import Model

from ..gear.shield import InputShield, OutputShield
from ..gear.sword import Sword, SwordResult, function_sword
from ..models.settings import ModelSettings
from ..util._print import transform_string_function_style
from ..util._types import MaybeAwaitable, RunContextWrapper, TContext

########################################################
#                    Private Methods
########################################################


def _create_agent_sword(
    agent: Agent[TContext],
    sword_name: str | None,
    sword_description: str | None,
    custom_output_extractor: Callable[[Any], str] | None = None,
) -> Sword:
    @function_sword(
        name_override=sword_name or transform_string_function_style(agent.name),
        description_override=sword_description or "",
    )
    async def _agent_sword(ctx: RunContextWrapper[TContext], input: str) -> Any:
        return await agent.run(ctx, input, custom_output_extractor)

    return _agent_sword


########################################################
#            Data Classes for Swords
########################################################


@dataclass
class SwordsToFinalOutputResult:
    is_final_output: bool
    final_output: Any | None = None


SwordsToFinalOutputFunction: TypeAlias = Callable[
    [RunContextWrapper[TContext], list[SwordResult]],
    MaybeAwaitable[SwordsToFinalOutputResult],
]


########################################################
#            Main Data Class for Agents
########################################################


@dataclass
class Agent(Generic[TContext]):
    """AI agent with swords, shields, orbs, and charms."""

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
    swords: list[Sword] = field(default_factory=list)
    input_shields: list[InputShield[TContext]] = field(default_factory=list)
    output_shields: list[OutputShield[TContext]] = field(default_factory=list)
    output_type: type[Any] | None = None
    charms: AgentCharms[TContext] | None = None
    sword_use_behavior: (
        Literal["run_llm_again", "stop_on_first_sword"]
        | dict[Literal["stop_at_sword_names"], list[str]]
    ) = "run_llm_again"

    def clone(self, **kwargs: Any) -> Agent[TContext]:
        """Create a copy of the agent with optional field updates."""
        return dataclasses.replace(self, **kwargs)

    def as_sword(
        self,
        sword_name: str | None = None,
        sword_description: str | None = None,
        custom_output_extractor: Callable[[Any], str] | None = None,
    ) -> Sword:
        """Converts agent to sword for other agents."""
        return _create_agent_sword(self, sword_name, sword_description, custom_output_extractor)

    async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
        """Get the system prompt for the agent."""
        if self.instructions is None:
            return None

        if isinstance(self.instructions, str):
            return self.instructions

        if not callable(self.instructions):
            raise TypeError(f"Invalid instructions type: {type(self.instructions)}")

        result = self.instructions(run_context, self)
        if inspect.iscoroutinefunction(self.instructions):
            return await cast(Awaitable[str], result)
        return cast(str, result)
