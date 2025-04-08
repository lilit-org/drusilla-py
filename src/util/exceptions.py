"""
This module defines custom exceptions used throughout the framework.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..gear.shield import InputShieldResult, OutputShieldResult


################################################
#         Exception classes
################################################


class AgentError(Exception):
    """Base class for all agent-related exceptions."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class AgentExecutionError(AgentError):
    """Raised when an error occurs during agent execution."""


class GenericError(AgentError):
    """Raised when an unexpected error occurs during agent execution."""


class MessageError(AgentError):
    """Raised for exceptions with a message attribute."""


class ConnectionError(AgentError):
    """Raised for exceptions related to network connections."""


class RunnerError(MessageError):
    """Raised for exceptions with a message attribute."""


class ModelError(MessageError):
    """Raised when model acts unexpectedly (e.g. invalid sword calls or malformed JSON)."""


class MaxTurnsError(MessageError):
    """Raised when max turns limit is reached."""


class UsageError(MessageError):
    """Raised for usage errors."""


class InputShieldError(AgentError):
    """Raised when an input shield's tripwire is triggered."""

    def __init__(self, result: "InputShieldResult") -> None:
        self.result = result
        shield_name = result.shield.name or "unnamed"
        super().__init__(f"Input shield {shield_name} triggered")


class OutputShieldError(AgentError):
    """Raised when an output shield's tripwire is triggered."""

    def __init__(self, result: "OutputShieldResult") -> None:
        self.result = result
        shield_name = result.shield.name or "unnamed"
        super().__init__(f"Output shield {shield_name} triggered")


################################################
#        Public utility functions
################################################


def format_error_message(error: Exception, message_template: str, context: Any = None) -> str:
    """Format an error message using a template string."""

    if context is None:
        template = message_template.replace("{context}", "None")
        return template.format(error=str(error))

    return message_template.format(error=str(error), context=context)


def create_error_handler(message_template: str) -> Callable[[Any, Exception], str]:
    """Create a context-aware error handler function."""

    def error_handler(context: Any, error: Exception) -> str:
        return format_error_message(error, message_template, context)

    return error_handler
