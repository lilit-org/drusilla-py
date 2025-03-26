from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

########################################################
#            Data class for settings                   #
########################################################

@dataclass
class ModelSettings:
    """Optional LLM configuration parameters (temperature, top_p, penalties, etc.).
    Check provider API docs for supported parameters."""

    temperature: float | None = None
    """Sampling temperature for model outputs."""

    top_p: float | None = None
    """Nucleus sampling parameter."""

    frequency_penalty: float | None = None
    """Penalty for token frequency."""

    presence_penalty: float | None = None
    """Penalty for token presence."""

    tool_choice: Literal["auto", "required", "none"] | str | None = None
    """Tool selection strategy."""

    parallel_tool_calls: bool | None = False
    """Enable parallel tool execution."""

    truncation: Literal["auto", "disabled"] | None = None
    """Output truncation strategy."""

    max_tokens: int | None = None
    """Maximum output token limit."""

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
