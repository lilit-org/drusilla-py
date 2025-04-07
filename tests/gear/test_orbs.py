"""Fixed tests for the orbs module."""

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
def mock_agent():
    """Fixture for creating a mock agent."""
    return MockAgent()


@pytest.fixture
def mock_context():
    """Fixture for creating a mock context."""
    return RunContextWrapper(context={}, usage=None)


@pytest.mark.asyncio
async def test_orbs_without_input(mock_agent: MockAgent, mock_context: RunContextWrapper[Any]):
    """Test creating orbs without input."""

    async def test_orbs(ctx: RunContextWrapper[Any]) -> None:
        pass

    # Create orbs instance
    orb = Orbs(
        on_invoke_orbs=lambda ctx, input_json=None: mock_agent,
        name="test_orbs",
        description="Test orbs description",
    )

    assert isinstance(orb, Orbs)
    assert orb.name == "test_orbs"
    assert orb.description == "Test orbs description"


@pytest.mark.asyncio
async def test_orbs_with_input(mock_agent: MockAgent, mock_context: RunContextWrapper[Any]):
    """Test creating orbs with input."""

    async def test_orbs(ctx: RunContextWrapper[Any], input_data: OrbsTestInput) -> None:
        assert ctx == mock_context
        assert input_data.message == "test message"

    # Create orbs instance with input schema
    orb = Orbs(
        on_invoke_orbs=lambda ctx, input_json=None: mock_agent,
        name="test_orbs",
        description="Test orbs description",
        input_json_schema=OrbsTestInput.model_json_schema(),
    )

    assert isinstance(orb, Orbs)
    assert orb.name == "test_orbs"
    assert orb.description == "Test orbs description"
    assert orb.input_json_schema is not None


def test_orbs_input_filter():
    """Test the input filter functionality."""

    def input_filter(data: OrbsInputData) -> OrbsInputData:
        return data

    orb = Orbs(
        on_invoke_orbs=lambda ctx, input_json=None: None,
        name="test_orbs",
        input_filter=input_filter,
    )

    assert isinstance(orb, Orbs)
    assert orb.input_filter == input_filter


def test_orbs_validation_errors(mock_agent):
    """Test validation errors in orbs creation."""

    class InvalidInput:
        """Invalid input class without model_json_schema."""

    async def test_orbs(ctx: RunContextWrapper[Any], input_data: InvalidInput) -> None:
        pass

    # Add input_type to the function
    test_orbs.input_type = InvalidInput

    with pytest.raises(
        UsageError, match="input_type must be a Pydantic model with model_json_schema method"
    ):
        # Create orbs instance with invalid input type
        orbs(mock_agent)(test_orbs)
