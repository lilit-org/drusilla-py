from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools._guardrail import InputGuardrailResult, OutputGuardrailResult


class AgentError(Exception):
    """Base exception for Agents SDK."""


class ModelError(AgentError):
    """Raised when model acts unexpectedly (e.g. invalid tool calls or malformed JSON)."""

    message: str

    def __init__(self, message: str):
        self.message = message


class MaxTurnsError(AgentError):
    """Raised when max turns limit is reached."""

    message: str

    def __init__(self, message: str):
        self.message = message


class UsageError(AgentError):
    """Raised for SDK usage errors."""

    message: str

    def __init__(self, message: str):
        self.message = message


class InputGuardrailError(AgentError):
    """Raised when input guardrail tripwire is triggered."""

    guardrail_result: "InputGuardrailResult"
    """Triggered guardrail result data."""

    def __init__(self, guardrail_result: "InputGuardrailResult"):
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.__class__.__name__} triggered tripwire"
        )


class OutputGuardrailError(AgentError):
    """Raised when output guardrail tripwire is triggered."""

    guardrail_result: "OutputGuardrailResult"
    """Triggered guardrail result data."""

    def __init__(self, guardrail_result: "OutputGuardrailResult"):
        self.guardrail_result = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.__class__.__name__} triggered tripwire"
        )
