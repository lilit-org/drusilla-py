from dataclasses import dataclass, field

from ..gear.orbs import OrbsInputFilter
from ..gear.shield import InputShield, OutputShield
from ..models.interface import Model
from ..models.provider import ModelProvider
from ..models.settings import ModelSettings
from ..util.constants import MAX_TURNS
from ..util.types import TContext


@dataclass(frozen=True)
class RunConfig:
    """Configuration for agent execution."""

    model: str | Model | None = None
    model_provider: ModelProvider = field(default_factory=ModelProvider)
    model_settings: ModelSettings | None = None
    orbs_input_filter: OrbsInputFilter | None = None
    input_shields: list[InputShield[TContext]] = field(default_factory=list)
    output_shields: list[OutputShield[TContext]] = field(default_factory=list)
    max_turns: int = MAX_TURNS
