"""
The Agent module provides a powerful framework for creating
and managing AI agents with advanced capabilities.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

from pydantic import TypeAdapter

from src.util.exceptions import AgentExecutionError, UsageError
from src.util.schema import ensure_strict_json_schema, get_type_adapter, type_to_str
from src.util.types import MaybeAwaitable, RunContextWrapper, TContext

from ..util.constants import config, err

if TYPE_CHECKING:
    from src.gear.charms import AgentCharms
    from src.gear.orbs import Orbs
    from src.models.interface import Model

from src.gear.shield import InputShield, OutputShield
from src.gear.sword import Sword, function_sword
from src.models.settings import ModelSettings
from src.util.print import transform_string_function_style, validate_json

########################################################
#                    Private Methods
########################################################


def _create_agent_sword(
    agent: AgentV1[TContext],
    sword_name: str | None,
    sword_description: str | None,
    custom_output_extractor: Callable[[Any], str] | None = None,
) -> Sword:
    @function_sword(
        name_override=sword_name or transform_string_function_style(agent.name),
        description_override=sword_description or "",
    )
    async def _agent_sword(ctx: RunContextWrapper[TContext], input: str) -> Any:
        from ..runners.run import Runner

        result = await Runner.run(
            starting_agent=agent, input=input, context=ctx, max_turns=config.MAX_TURNS
        )
        if custom_output_extractor:
            return custom_output_extractor(result.final_output)
        return result.final_output

    return _agent_sword


########################################################
#            Main Data Class for Agents
########################################################


@dataclass
class AgentV1(Generic[TContext]):
    """AI agent with swords, shields, orbs, and charms."""

    name: str
    instructions: (
        str
        | Callable[
            [RunContextWrapper[TContext], AgentV1[TContext]],
            MaybeAwaitable[str],
        ]
        | None
    ) = None

    orbs_description: str | None = None
    orbs: list[AgentV1[Any] | Orbs[TContext]] = field(default_factory=list)
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

    def clone(self, **kwargs: Any) -> AgentV1[TContext]:
        """Create a copy of the agent with optional field updates."""
        return replace(self, **kwargs)

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
            error_msg = err.AGENT_EXEC_ERROR.format(
                error=f"Invalid instructions type: {type(self.instructions)}"
            )
            raise AgentExecutionError(error_msg)

        result = self.instructions(run_context, self)
        if inspect.iscoroutinefunction(self.instructions):
            return await cast(Awaitable[str], result)
        return cast(str, result)


########################################################
#             Main Class for Output Schema            #
########################################################


@dataclass(init=False)
class AgentV1OutputSchema:
    """Schema for validating and parsing LLM output into specified types."""

    output_type: type[Any]
    strict_json_schema: bool
    _type_adapter: TypeAdapter[Any]
    _is_plain_text: bool

    def __init__(self, output_type: type[Any], strict_json_schema: bool = True):
        """
        Args:
            output_type: Type to validate against
            strict_json_schema: Enable strict validation (recommended)
        """
        self.output_type = output_type
        self.strict_json_schema = strict_json_schema
        self._is_plain_text = output_type is None or output_type is str
        self._type_adapter = get_type_adapter(output_type)

    def is_plain_text(self) -> bool:
        return self._is_plain_text

    def json_schema(self) -> dict[str, Any]:
        if self.is_plain_text():
            raise UsageError(err.USAGE_ERROR.format(error="No JSON schema for plain text output"))
        schema = self._type_adapter.json_schema()
        if self.strict_json_schema:
            schema = ensure_strict_json_schema(schema)
        return schema

    def validate_json(self, json_str: str, partial: bool = False) -> Any:
        return validate_json(json_str, self._type_adapter, partial)

    def output_type_name(self) -> str:
        return type_to_str(self.output_type)
