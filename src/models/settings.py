"""
Model Configuration Management

This module provides the ModelSettings class, a configuration manager for
fine-tuning model behavior. It handles critical model parameters such as
temperature, top-p sampling, and token limits, with built-in validation to
ensure parameters stay within safe operating ranges.

The class supports:
- Temperature control for response creativity (0.0 to 2.0)
- Top-p sampling for response diversity (0.0 to 1.0)
- Maximum token limits for response length
- Sword choice configuration for function calling
- Parallel sword call optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..util.exceptions import UsageError

########################################################
#            Data class for settings                   #
########################################################


@dataclass
class ModelSettings:

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    sword_choice: Literal["auto", "required", "none"] | None = None
    parallel_sword_calls: bool | None = None

    def resolve(self, other: ModelSettings | None) -> ModelSettings:
        """Resolve settings with another instance, preferring non-None values."""
        if other is None:
            return self
        return ModelSettings(
            temperature=other.temperature if other.temperature is not None else self.temperature,
            top_p=other.top_p if other.top_p is not None else self.top_p,
            max_tokens=other.max_tokens if other.max_tokens is not None else self.max_tokens,
            sword_choice=(
                other.sword_choice if other.sword_choice is not None else self.sword_choice
            ),
            parallel_sword_calls=(
                other.parallel_sword_calls
                if other.parallel_sword_calls is not None
                else self.parallel_sword_calls
            ),
        )

    def validate(self) -> None:
        """Validate model settings."""
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise UsageError("temperature must be between 0 and 2")
        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise UsageError("top_p must be between 0 and 1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise UsageError("max_tokens must be at least 1")
