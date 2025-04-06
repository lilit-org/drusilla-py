from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..util._exceptions import UsageError

########################################################
#            Data class for settings                   #
########################################################


@dataclass(frozen=True)
class ModelSettings:
    """Optional LLM configuration parameters (temperature, top_p, penalties, etc.)."""

    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    sword_choice: Literal["auto", "required", "none"] | str | None = None
    parallel_sword_calls: bool | None = False
    truncation: Literal["auto", "disabled"] | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate numeric parameters after initialization."""
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise UsageError("temperature must be between 0 and 2")
        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise UsageError("top_p must be between 0 and 1")
        if self.frequency_penalty is not None and not -2 <= self.frequency_penalty <= 2:
            raise UsageError("frequency_penalty must be between -2 and 2")
        if self.presence_penalty is not None and not -2 <= self.presence_penalty <= 2:
            raise UsageError("presence_penalty must be between -2 and 2")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise UsageError("max_tokens must be positive")

    def resolve(self, override: ModelSettings | None) -> ModelSettings:
        """Merge override settings with current settings."""
        if not override:
            return self
        return replace(self, **{k: v for k, v in override.__dict__.items() if v is not None})
