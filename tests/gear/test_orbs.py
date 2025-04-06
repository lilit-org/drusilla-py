"""Tests for the orbs module."""

from typing import Any

import pytest
from pydantic import BaseModel

from src.agents.agent import Agent
from src.gear.orbs import Orbs, OrbsInputData, orbs
from src.util._types import RunContextWrapper
from src.util.exceptions import UsageError


class OrbsTestInput(BaseModel):
    """Test input model for orbs."""

    message: str


class MockAgent(Agent[Any]):
    """Mock agent for testing."""

    def __init__(self, name: str = "test_agent"):
        self.name = name
        self.orbs_description = "Test agent description"


@pytest.fixture
def mock_agent() -> MockAgent:
    """Fixture for creating a mock agent."""
    return MockAgent()


@pytest.fixture
def mock_context() -> RunContextWrapper[Any]:
    """Fixture for creating a mock context."""
    return RunContextWrapper(context={})


def test_orbs_default_sword_name(mock_agent: MockAgent):
    """Test the default sword name generation."""
    expected_name = "transfer_to_test_agent"
    assert Orbs.default_sword_name(mock_agent) == expected_name


def test_orbs_default_sword_description(mock_agent: MockAgent):
    """Test the default sword description generation."""
    expected_description = (
        "Orbs to the test_agent agent to handle the request. Test agent description"
    )
    assert Orbs.default_sword_description(mock_agent) == expected_description


@pytest.mark.asyncio
async def test_orbs_without_input(mock_agent: MockAgent, mock_context: RunContextWrapper[Any]):
    """Test creating orbs without input."""
    orb = orbs(
        mock_agent,
    )

    assert orb.sword_name == "transfer_to_test_agent"
    assert (
        orb.sword_description
        == "Orbs to the test_agent agent to handle the request. Test agent description"
    )
    assert orb.input_json_schema == {
        "additionalProperties": False,
        "properties": {},
        "required": [],
        "type": "object",
    }
    assert orb.agent_name == "test_agent"
    assert orb.input_filter is None

    # Test invoking the orbs
    result = await orb.on_invoke_orbs(mock_context)
    assert result == mock_agent


@pytest.mark.asyncio
async def test_orbs_with_input(mock_agent: MockAgent, mock_context: RunContextWrapper[Any]):
    """Test creating orbs with input."""

    async def on_orbs(ctx: RunContextWrapper[Any], input_data: OrbsTestInput) -> None:
        assert ctx == mock_context
        assert input_data.message == "test message"

    orb = orbs(
        mock_agent,
        on_orbs=on_orbs,
        input_type=OrbsTestInput,
    )

    assert orb.sword_name == "transfer_to_test_agent"
    assert (
        orb.sword_description
        == "Orbs to the test_agent agent to handle the request. Test agent description"
    )
    assert "message" in orb.input_json_schema["properties"]
    assert orb.agent_name == "test_agent"
    assert orb.input_filter is None

    # Test invoking the orbs
    result = await orb.on_invoke_orbs(mock_context, '{"message": "test message"}')
    assert result == mock_agent


def test_orbs_input_filter():
    """Test the input filter functionality."""

    def input_filter(data: OrbsInputData) -> OrbsInputData:
        return data

    orb = orbs(
        MockAgent(),
        input_filter=input_filter,
    )

    assert orb.input_filter == input_filter


def test_orbs_validation_errors():
    """Test validation errors in orbs creation."""
    with pytest.raises(
        UsageError, match="You must provide either both on_input and input_type, or neither"
    ):
        # Should raise error when only one of on_orbs or input_type is provided
        orbs(
            MockAgent(),
            on_orbs=lambda ctx: None,
            input_type=None,
        )

    with pytest.raises(UsageError, match="on_orbs must take one argument: context"):
        # Should raise error when on_orbs takes wrong number of arguments
        orbs(
            MockAgent(),
            on_orbs=lambda ctx, x, y: None,
        )
