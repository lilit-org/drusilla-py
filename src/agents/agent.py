"""
This module implements the Agent class, which provides a
flexible framework for creating decentralized AI agents.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, cast

from ..gear.orbs import Orbs
from ..gear.shields import InputShield, OutputShield
from ..gear.swords import FunctionSword, FunctionSwordResult, Sword, function_sword
from ..models.settings import ModelSettings
from ..util import _json
from ..util._run_context import RunContextWrapper, TContext
from ..util._types import MaybeAwaitable

if TYPE_CHECKING:
    from ..models.interface import Model
    from ..util._lifecycle import AgentHooks


########################################################
#                    Private Methods
########################################################


def _create_agent_sword(
    agent: Agent[TContext],
    sword_name: str | None,
    sword_description: str | None,
    custom_output_extractor: Callable[[Any], str] | None = None,
) -> FunctionSword:
    @function_sword(
        name_override=sword_name or _json.transform_string_function_style(agent.name),
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
    [RunContextWrapper[TContext], list[FunctionSwordResult]],
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
    hooks: AgentHooks[TContext] | None = None
    sword_use_behavior: (
        Literal["run_llm_again", "stop_on_first_sword"]
        | dict[Literal["stop_at_sword_names"], list[str]]
    ) = "run_llm_again"

    def clone(self, **kwargs: Any) -> Agent[TContext]:
        """Create a copy of the agent with optional field updates."""
        defaults = {
            "orbs": [],
            "model": "",
            "model_settings": ModelSettings(),
            "swords": [],
            "input_shields": [],
            "output_shields": [],
            "output_type": None,
            "hooks": None,
            "sword_use_behavior": "run_llm_again",
        }
        defaults.update(kwargs)
        return dataclasses.replace(self, **defaults)

    def as_sword(
        self,
        sword_name: str | None,
        sword_description: str | None,
        custom_output_extractor: Callable[[Any], str] | None = None,
    ) -> FunctionSword:
        """Converts agent to sword for other agents."""
        return _create_agent_sword(self, sword_name, sword_description, custom_output_extractor)

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
