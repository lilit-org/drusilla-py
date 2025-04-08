from dataclasses import dataclass

import pytest

from src.util.exceptions import (
    AgentError,
    AgentExecutionError,
    GenericError,
    InputShieldError,
    MaxTurnsError,
    MessageError,
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
    assert exc_info.value.message == "Test error"


def test_generic_error():
    """Test GenericError class."""
    with pytest.raises(GenericError) as exc_info:
        raise GenericError("Test error")
    assert str(exc_info.value) == "Test error"
    assert exc_info.value.message == "Test error"


def test_message_error():
    """Test MessageError class."""
    with pytest.raises(MessageError) as exc_info:
        raise MessageError("Test message")
    assert str(exc_info.value) == "Test message"
    assert exc_info.value.message == "Test message"


def test_runner_error():
    """Test RunnerError class."""
    with pytest.raises(RunnerError) as exc_info:
        raise RunnerError("Test runner error")
    assert str(exc_info.value) == "Test runner error"
    assert exc_info.value.message == "Test runner error"
    assert isinstance(exc_info.value, MessageError)


def test_model_error():
    """Test ModelError class."""
    with pytest.raises(ModelError) as exc_info:
        raise ModelError("Invalid model response")
    assert str(exc_info.value) == "Invalid model response"
    assert exc_info.value.message == "Invalid model response"
    assert isinstance(exc_info.value, MessageError)


def test_max_turns_error():
    """Test MaxTurnsError class."""
    with pytest.raises(MaxTurnsError) as exc_info:
        raise MaxTurnsError("Maximum turns reached")
    assert str(exc_info.value) == "Maximum turns reached"
    assert exc_info.value.message == "Maximum turns reached"
    assert isinstance(exc_info.value, MessageError)


def test_usage_error():
    """Test UsageError class."""
    with pytest.raises(UsageError) as exc_info:
        raise UsageError("Invalid usage")
    assert str(exc_info.value) == "Invalid usage"
    assert exc_info.value.message == "Invalid usage"
    assert isinstance(exc_info.value, MessageError)


def test_input_shield_error():
    """Test InputShieldError class."""
    shield = MockShield(name="test_shield")
    result = MockShieldResult(shield=shield)

    with pytest.raises(InputShieldError) as exc_info:
        raise InputShieldError(result)

    assert str(exc_info.value) == "Input shield test_shield triggered"
    assert exc_info.value.result == result
    assert isinstance(exc_info.value, AgentError)
    assert exc_info.value.message == "Input shield test_shield triggered"


def test_input_shield_error_unnamed():
    """Test InputShieldError with unnamed shield."""
    shield = MockShield(name=None)
    result = MockShieldResult(shield=shield)

    with pytest.raises(InputShieldError) as exc_info:
        raise InputShieldError(result)

    assert str(exc_info.value) == "Input shield unnamed triggered"
    assert exc_info.value.result == result
    assert exc_info.value.message == "Input shield unnamed triggered"


def test_output_shield_error():
    """Test OutputShieldError class."""
    shield = MockShield(name="test_shield")
    result = MockShieldResult(shield=shield)

    with pytest.raises(OutputShieldError) as exc_info:
        raise OutputShieldError(result)

    assert str(exc_info.value) == "Output shield test_shield triggered"
    assert exc_info.value.result == result
    assert isinstance(exc_info.value, AgentError)
    assert exc_info.value.message == "Output shield test_shield triggered"


def test_output_shield_error_unnamed():
    """Test OutputShieldError with unnamed shield."""
    shield = MockShield(name=None)
    result = MockShieldResult(shield=shield)

    with pytest.raises(OutputShieldError) as exc_info:
        raise OutputShieldError(result)

    assert str(exc_info.value) == "Output shield unnamed triggered"
    assert exc_info.value.result == result
    assert exc_info.value.message == "Output shield unnamed triggered"


def test_agent_execution_error():
    """Test AgentExecutionError class."""
    with pytest.raises(AgentExecutionError) as exc_info:
        raise AgentExecutionError("Test execution error")
    assert str(exc_info.value) == "Test execution error"
    assert exc_info.value.message == "Test execution error"
    assert isinstance(exc_info.value, AgentError)


def test_exception_hierarchy():
    """Test exception class hierarchy."""
    # Test MessageError hierarchy
    assert issubclass(ModelError, MessageError)
    assert issubclass(MaxTurnsError, MessageError)
    assert issubclass(UsageError, MessageError)
    assert issubclass(RunnerError, MessageError)

    # Test AgentError hierarchy
    assert issubclass(MessageError, AgentError)
    assert issubclass(GenericError, AgentError)
    assert issubclass(AgentExecutionError, AgentError)
    assert issubclass(InputShieldError, AgentError)
    assert issubclass(OutputShieldError, AgentError)


def test_format_error_message():
    """Test format_error_message function."""
    error = ValueError("Test error")
    template = "An error occurred: {error}"
    result = format_error_message(error, template)
    assert result == "An error occurred: Test error"


def test_create_error_handler():
    """Test create_error_handler function."""
    template = "Handler error: {error}"
    handler = create_error_handler(template)

    # Test with a simple error
    error = ValueError("Test error")
    result = handler(None, error)
    assert result == "Handler error: Test error"

    # Test with a different error
    error = TypeError("Another error")
    result = handler(None, error)
    assert result == "Handler error: Another error"


def test_format_error_message_edge_cases():
    """Test format_error_message function with edge cases."""
    # Test with empty error message
    error = ValueError("")
    template = "Error: {error}"
    result = format_error_message(error, template)
    assert result == "Error: "

    # Test with error that has no message
    class CustomError(Exception):
        pass

    error = CustomError()
    result = format_error_message(error, template)
    assert result == "Error: "

    # Test with error that has custom __str__ method
    class CustomStrError(Exception):
        def __str__(self):
            return "Custom string representation"

    error = CustomStrError()
    result = format_error_message(error, template)
    assert result == "Error: Custom string representation"

    # Test with template that doesn't use the error
    template = "Static message"
    result = format_error_message(error, template)
    assert result == "Static message"


def test_create_error_handler_edge_cases():
    """Test create_error_handler function with edge cases."""
    # Test with empty template
    handler = create_error_handler("")
    error = ValueError("Test error")
    result = handler(None, error)
    assert result == ""

    # Test with template that has multiple error placeholders
    template = "Error: {error} - Details: {error}"
    handler = create_error_handler(template)
    result = handler(None, error)
    assert result == "Error: Test error - Details: Test error"

    # Test with different context objects
    class Context:
        def __init__(self, value):
            self.value = value

    context = Context("test")
    handler = create_error_handler("Error: {error} - Context: {context.value}")
    result = handler(context, error)
    assert result == "Error: Test error - Context: test"

    # Test with context that has no value attribute
    handler = create_error_handler("Error: {error} - Context: {context}")
    result = handler(None, error)
    assert result == "Error: Test error - Context: None"

    # Test with context that has custom __str__ method
    class CustomContext:
        def __str__(self):
            return "custom context"

    custom_context = CustomContext()
    handler = create_error_handler("Error: {error} - Context: {context}")
    result = handler(custom_context, error)
    assert result == "Error: Test error - Context: custom context"

    # Test with nested context attributes
    class NestedContext:
        def __init__(self, inner):
            self.inner = inner

    inner_context = Context("inner")
    nested_context = NestedContext(inner_context)
    handler = create_error_handler("Error: {error} - Context: {context.inner.value}")
    result = handler(nested_context, error)
    assert result == "Error: Test error - Context: inner"
