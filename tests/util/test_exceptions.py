from dataclasses import dataclass

import pytest

from src.util.exceptions import (
    AgentError,
    AgentExecutionError,
    GenericError,
    InputShieldError,
    MaxTurnsError,
    ModelError,
    OutputShieldError,
    RunnerError,
    UsageError,
    create_error_handler,
    format_error_message,
)


# Mock classes for testing
@dataclass
class MockShield:
    name: str | None


@dataclass
class MockShieldResult:
    shield: MockShield


def test_agent_error():
    """Test base AgentError class."""
    with pytest.raises(AgentError) as exc_info:
        raise AgentError("Test error")
    assert str(exc_info.value) == "Test error"


def test_generic_error():
    """Test GenericError class."""
    with pytest.raises(GenericError) as exc_info:
        raise GenericError("Test error")
    assert str(exc_info.value) == "Test error"


def test_runner_error():
    """Test RunnerError class."""
    with pytest.raises(RunnerError) as exc_info:
        raise RunnerError("Test runner error")
    assert str(exc_info.value) == "Test runner error"
    assert isinstance(exc_info.value, AgentError)


def test_model_error():
    """Test ModelError class."""
    with pytest.raises(ModelError) as exc_info:
        raise ModelError("Invalid model response")
    assert str(exc_info.value) == "Invalid model response"
    assert isinstance(exc_info.value, AgentError)


def test_max_turns_error():
    """Test MaxTurnsError class."""
    with pytest.raises(MaxTurnsError) as exc_info:
        raise MaxTurnsError("Maximum turns reached")
    assert str(exc_info.value) == "Maximum turns reached"
    assert isinstance(exc_info.value, AgentError)


def test_usage_error():
    """Test UsageError class."""
    with pytest.raises(UsageError) as exc_info:
        raise UsageError("Invalid usage")
    assert str(exc_info.value) == "Invalid usage"
    assert isinstance(exc_info.value, AgentError)


def test_shield_errors():
    """Test shield-related errors."""
    # Test InputShieldError with named shield
    shield = MockShield(name="test_shield")
    result = MockShieldResult(shield=shield)

    with pytest.raises(InputShieldError) as exc_info:
        raise InputShieldError(result)
    assert str(exc_info.value) == "Input shield test_shield triggered"
    assert exc_info.value.result == result
    assert isinstance(exc_info.value, AgentError)

    # Test InputShieldError with unnamed shield
    shield = MockShield(name=None)
    result = MockShieldResult(shield=shield)

    with pytest.raises(InputShieldError) as exc_info:
        raise InputShieldError(result)
    assert str(exc_info.value) == "Input shield unnamed triggered"
    assert exc_info.value.result == result

    # Test OutputShieldError with named shield
    shield = MockShield(name="test_shield")
    result = MockShieldResult(shield=shield)

    with pytest.raises(OutputShieldError) as exc_info:
        raise OutputShieldError(result)
    assert str(exc_info.value) == "Output shield test_shield triggered"
    assert exc_info.value.result == result
    assert isinstance(exc_info.value, AgentError)

    # Test OutputShieldError with unnamed shield
    shield = MockShield(name=None)
    result = MockShieldResult(shield=shield)

    with pytest.raises(OutputShieldError) as exc_info:
        raise OutputShieldError(result)
    assert str(exc_info.value) == "Output shield unnamed triggered"
    assert exc_info.value.result == result


def test_agent_execution_error():
    """Test AgentExecutionError class."""
    with pytest.raises(AgentExecutionError) as exc_info:
        raise AgentExecutionError("Test execution error")
    assert str(exc_info.value) == "Test execution error"
    assert isinstance(exc_info.value, AgentError)


def test_exception_hierarchy():
    """Test exception class hierarchy."""
    assert issubclass(ModelError, AgentError)
    assert issubclass(MaxTurnsError, AgentError)
    assert issubclass(UsageError, AgentError)
    assert issubclass(RunnerError, AgentError)
    assert issubclass(GenericError, AgentError)
    assert issubclass(AgentExecutionError, AgentError)
    assert issubclass(InputShieldError, AgentError)
    assert issubclass(OutputShieldError, AgentError)


def test_format_error_message():
    """Test format_error_message function."""
    # Test basic error formatting
    error = ValueError("Test error")
    assert (
        format_error_message(error, "An error occurred: {error}") == "An error occurred: Test error"
    )

    # Test empty error message
    error = ValueError("")
    assert format_error_message(error, "Error: {error}") == "Error: "

    # Test error with no message
    class CustomError(Exception):
        pass

    error = CustomError()
    assert format_error_message(error, "Error: {error}") == "Error: "

    # Test error with custom __str__
    class CustomStrError(Exception):
        def __str__(self):
            return "Custom string representation"

    error = CustomStrError()
    assert format_error_message(error, "Error: {error}") == "Error: Custom string representation"

    # Test static message
    assert format_error_message(error, "Static message") == "Static message"


def test_create_error_handler():
    """Test create_error_handler function."""
    # Test basic error handling
    handler = create_error_handler("Handler error: {error}")
    error = ValueError("Test error")
    assert handler(None, error) == "Handler error: Test error"

    # Test empty template
    handler = create_error_handler("")
    assert handler(None, error) == ""

    # Test multiple error placeholders
    handler = create_error_handler("Error: {error} - Details: {error}")
    assert handler(None, error) == "Error: Test error - Details: Test error"

    # Test with context
    class Context:
        def __init__(self, value):
            self.value = value

    context = Context("test")
    handler = create_error_handler("Error: {error} - Context: {context.value}")
    assert handler(context, error) == "Error: Test error - Context: test"

    # Test with custom context __str__
    class CustomContext:
        def __str__(self):
            return "custom context"

    custom_context = CustomContext()
    handler = create_error_handler("Error: {error} - Context: {context}")
    assert handler(custom_context, error) == "Error: Test error - Context: custom context"

    # Test with nested context
    class NestedContext:
        def __init__(self, inner):
            self.inner = inner

    inner_context = Context("inner")
    nested_context = NestedContext(inner_context)
    handler = create_error_handler("Error: {error} - Context: {context.inner.value}")
    assert handler(nested_context, error) == "Error: Test error - Context: inner"
