"""Tests for the orbs module."""

from typing import Any

import pytest
from pydantic import BaseModel

from src.agents.agent import Agent
from src.gear.orbs import Orbs, OrbsInputData, orbs
from src.util._exceptions import UsageError
from src.util._types import RunContextWrapper


class OrbsTestInput(BaseModel):
    """Test input model for orbs."""

    message: str


class MockAgent(Agent[Any]):
    """Mock agent for testing."""

    def __init__(self, name: str = "test_agent"):
        super().__init__(name=name)
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
    assert Orbs.default_name(mock_agent) == expected_name


def test_orbs_default_sword_description(mock_agent: MockAgent):
    """Test the default sword description generation."""
    expected_description = (
        "Orbs to the test_agent agent to handle the request. Test agent description"
    )
    assert Orbs.default_description(mock_agent) == expected_description


@pytest.mark.asyncio
async def test_orbs_without_input(mock_agent: MockAgent, mock_context: RunContextWrapper[Any]):
    """Test creating orbs without input."""

    @orbs
    async def test_orbs(ctx: RunContextWrapper[Any]) -> None:
        pass

    orb = test_orbs(mock_agent)
    assert orb.name == "transfer_to_test_agent"


@pytest.mark.asyncio
async def test_orbs_with_input(mock_agent: MockAgent, mock_context: RunContextWrapper[Any]):
    """Test creating orbs with input."""

    @orbs(input_type=OrbsTestInput)
    async def test_orbs(ctx: RunContextWrapper[Any], input_data: OrbsTestInput) -> None:
        assert ctx == mock_context
        assert input_data.message == "test message"

    orb = test_orbs(mock_agent)
    assert orb.name == "transfer_to_test_agent"
    assert orb.input_json_schema is not None


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
    with pytest.raises(UsageError, match="on_orbs must take two arguments: context and input"):

        @orbs(input_type=str)
        async def test_orbs(ctx: RunContextWrapper[Any], x: Any, y: Any) -> None:
            pass

    with pytest.raises(
        UsageError,
        match="You must provide either both on_input and input_type, or neither",
    ):

        @orbs(input_type=None)
        async def test_orbs(ctx: RunContextWrapper[Any]) -> None:
            pass
