"""
This module defines custom exceptions used throughout the Agents SDK.
It includes:

1. Base exception classes for:
   - Agent-related errors
   - Model-related errors
   - Usage-related errors
   - Input shield errors
   - Output shield errors

2. Specific exception types for:
   - Agent execution errors
   - Generic errors
   - Message errors
   - Model errors
   - Max turns errors
   - Usage errors
   - Input shield errors
   - Output shield errors
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..gear.shield import InputShieldResult, OutputShieldResult


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


def format_error_message(error: Exception, message_template: str) -> str:
    """Format an error message using a template string.

    Args:
        error: The exception that occurred
        message_template: A string template that can contain {error} placeholder

    Returns:
        Formatted error message string
    """
    return message_template.format(error=str(error))


def create_error_handler(message_template: str) -> Callable[[Any, Exception], str]:
    """Create a context-aware error handler function.

    Args:
        message_template: A string template that can contain {error} placeholder

    Returns:
        A function that takes a context and error and returns a formatted message
    """

    def error_handler(_: Any, error: Exception) -> str:
        return format_error_message(error, message_template)

    return error_handler
