from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..gear.shields import InputShieldResult, OutputShieldResult


class AgentError(Exception):
    """Base exception for Agents SDK."""


class AgentExecutionError(AgentError):
    """Raised when an error occurs during agent execution."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        super().__init__(f"Agent execution failed: {str(error)}")


class GenericError(AgentError):
    """Raised when an unexpected error occurs during agent execution."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        super().__init__(str(error))


class MessageError(AgentError):
    """Base class for exceptions with a message attribute."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ModelError(MessageError):
    """Raised when model acts unexpectedly (e.g. invalid sword calls or malformed JSON)."""


class MaxTurnsError(MessageError):
    """Raised when max turns limit is reached."""


class UsageError(MessageError):
    """Raised for SDK usage errors."""


class InputShieldError(AgentError):
    """Raised when an input shield's tripwire is triggered."""

    def __init__(self, result: "InputShieldResult") -> None:
        self.result = result
        super().__init__(f"Input shield {result.shield.name or 'unnamed'} triggered")


class OutputShieldError(AgentError):
    """Raised when an output shield's tripwire is triggered."""

    def __init__(self, result: "OutputShieldResult") -> None:
        self.result = result
        super().__init__(f"Output shield {result.shield.name or 'unnamed'} triggered")
