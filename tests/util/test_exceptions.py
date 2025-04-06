from dataclasses import dataclass

import pytest

from src.util._exceptions import (
    AgentError,
    AgentExecutionError,
    GenericError,
    InputShieldError,
    MaxTurnsError,
    MessageError,
    ModelError,
    OutputShieldError,
    UsageError,
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


def test_agent_execution_error():
    """Test AgentExecutionError class."""
    original_error = ValueError("Original error")
    with pytest.raises(AgentExecutionError) as exc_info:
        raise AgentExecutionError(original_error)

    assert str(exc_info.value) == "Agent execution failed: Original error"
    assert exc_info.value.error == original_error


def test_generic_error():
    """Test GenericError class."""
    original_error = ValueError("Original error")
    with pytest.raises(GenericError) as exc_info:
        raise GenericError(original_error)

    assert str(exc_info.value) == "Original error"
    assert exc_info.value.error == original_error


def test_message_error():
    """Test MessageError class."""
    with pytest.raises(MessageError) as exc_info:
        raise MessageError("Test message")

    assert str(exc_info.value) == "Test message"
    assert exc_info.value.message == "Test message"


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


def test_input_shield_error_unnamed():
    """Test InputShieldError with unnamed shield."""
    shield = MockShield(name=None)
    result = MockShieldResult(shield=shield)

    with pytest.raises(InputShieldError) as exc_info:
        raise InputShieldError(result)

    assert str(exc_info.value) == "Input shield unnamed triggered"
    assert exc_info.value.result == result


def test_output_shield_error():
    """Test OutputShieldError class."""
    shield = MockShield(name="test_shield")
    result = MockShieldResult(shield=shield)

    with pytest.raises(OutputShieldError) as exc_info:
        raise OutputShieldError(result)

    assert str(exc_info.value) == "Output shield test_shield triggered"
    assert exc_info.value.result == result
    assert isinstance(exc_info.value, AgentError)


def test_output_shield_error_unnamed():
    """Test OutputShieldError with unnamed shield."""
    shield = MockShield(name=None)
    result = MockShieldResult(shield=shield)

    with pytest.raises(OutputShieldError) as exc_info:
        raise OutputShieldError(result)

    assert str(exc_info.value) == "Output shield unnamed triggered"
    assert exc_info.value.result == result


def test_exception_hierarchy():
    """Test exception class hierarchy."""
    # Test MessageError hierarchy
    assert issubclass(ModelError, MessageError)
    assert issubclass(MaxTurnsError, MessageError)
    assert issubclass(UsageError, MessageError)

    # Test AgentError hierarchy
    assert issubclass(MessageError, AgentError)
    assert issubclass(AgentExecutionError, AgentError)
    assert issubclass(GenericError, AgentError)
    assert issubclass(InputShieldError, AgentError)
    assert issubclass(OutputShieldError, AgentError)
