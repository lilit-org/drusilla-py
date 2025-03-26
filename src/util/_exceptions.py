from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._guardrail import InputGuardrailResult, OutputGuardrailResult


class AgentError(Exception):
    """Base exception for Agents SDK."""


class GenericError(AgentError):
    """Raised when an unexpected error occurs during agent execution."""

    def __init__(self, error: Exception) -> None:
        self.error: Exception = error
        super().__init__(str(error))


class MessageError(AgentError):
    """Base class for exceptions with a message attribute."""

    def __init__(self, message: str) -> None:
        self.message: str = message
        super().__init__(message)


class ModelError(MessageError):
    """Raised when model acts unexpectedly (e.g. invalid tool calls or malformed JSON)."""


class MaxTurnsError(MessageError):
    """Raised when max turns limit is reached."""


class UsageError(MessageError):
    """Raised for SDK usage errors."""


class InputGuardrailError(AgentError):
    """Raised when input guardrail tripwire is triggered."""

    def __init__(self, guardrail_result: "InputGuardrailResult") -> None:
        self.guardrail_result: InputGuardrailResult = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.__class__.__name__} triggered tripwire"
        )


class OutputGuardrailError(AgentError):
    """Raised when output guardrail tripwire is triggered."""

    def __init__(self, guardrail_result: "OutputGuardrailResult") -> None:
        self.guardrail_result: OutputGuardrailResult = guardrail_result
        super().__init__(
            f"Guardrail {guardrail_result.guardrail.__class__.__name__} triggered tripwire"
        )
