from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    tool_choice: Literal["auto", "required", "none"] | str | None = None
    parallel_tool_calls: bool | None = False
    truncation: Literal["auto", "disabled"] | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate numeric parameters after initialization."""
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.frequency_penalty is not None and not -2 <= self.frequency_penalty <= 2:
            raise ValueError("frequency_penalty must be between -2 and 2")
        if self.presence_penalty is not None and not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be between -2 and 2")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    def resolve(self, override: ModelSettings | None) -> ModelSettings:
        """Merge override settings with current settings, preferring non-None values from override."""
        if override is None:
            return self
        return ModelSettings(
            temperature=override.temperature or self.temperature,
            top_p=override.top_p or self.top_p,
            frequency_penalty=override.frequency_penalty or self.frequency_penalty,
            presence_penalty=override.presence_penalty or self.presence_penalty,
            tool_choice=override.tool_choice or self.tool_choice,
            parallel_tool_calls=override.parallel_tool_calls or self.parallel_tool_calls,
            truncation=override.truncation or self.truncation,
            max_tokens=override.max_tokens or self.max_tokens,
        )
