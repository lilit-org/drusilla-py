"""Tests for the orbs module."""

from typing import Any

import pytest
from pydantic import BaseModel

from src.agents.agent_v1 import AgentV1 as Agent
from src.gear.orbs import Orbs, OrbsInputData, orbs
from src.runners.items import MessageOutputItem
from src.util.constants import ERROR_MESSAGES
from src.util.exceptions import UsageError
from src.util.types import ResponseInputItemParam, RunContextWrapper


class OrbsTestInput(BaseModel):
    """Test input model for orbs."""

    message: str


class MockAgent(Agent[Any]):
    """Mock agent for testing."""

    def __init__(self, name: str = "test_agent", orbs_description: str = "Test agent description"):
        super().__init__(name=name)
        self.orbs_description = orbs_description


@pytest.fixture
def mock_agent():
    """Fixture for creating a mock agent."""
    return MockAgent()


@pytest.fixture
def mock_context():
    """Fixture for creating a mock context."""
    return RunContextWrapper(context={}, usage=None)


@pytest.fixture
def sample_input_data(mock_agent):
    """Fixture for creating sample input data."""
    input_item: ResponseInputItemParam = {
        "type": "message",
        "content": "test history",
        "role": "user",
    }
    return OrbsInputData(
        input_history=(input_item,),
        pre_orbs_items=(
            MessageOutputItem(
                agent=mock_agent,
                raw_item={
                    "type": "message",
                    "content": [{"type": "output_text", "text": "pre item"}],
                },
            ),
        ),
        new_items=(
            MessageOutputItem(
                agent=mock_agent,
                raw_item={
                    "type": "message",
                    "content": [{"type": "output_text", "text": "new item"}],
                },
            ),
        ),
    )


@pytest.mark.asyncio
async def test_orbs_creation_and_defaults(
    mock_agent: MockAgent, mock_context: RunContextWrapper[Any]
):
    """Test basic orbs creation and default values."""
    orb = Orbs(
        on_invoke_orbs=lambda ctx, input_json=None: mock_agent,
        name="test_orbs",
        description="Test orbs description",
    )

    assert isinstance(orb, Orbs)
    assert orb.name == "test_orbs"
    assert orb.description == "Test orbs description"
    assert orb.input_json_schema is None
    assert orb.input_filter is None

    # Test default name and description
    assert Orbs.default_name(mock_agent) == "transfer_to_test_agent"
    assert (
        Orbs.default_description(mock_agent)
        == "Orbs to the test_agent agent to handle the request. Test agent description"
    )


@pytest.mark.asyncio
async def test_orbs_with_input_schema(mock_agent: MockAgent, mock_context: RunContextWrapper[Any]):
    """Test orbs with input schema and validation."""

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

    assert orb.input_json_schema is not None
    assert orb.input_json_schema["title"] == "OrbsTestInput"


def test_orbs_input_data_creation(sample_input_data):
    """Test OrbsInputData creation and properties."""
    assert isinstance(sample_input_data, OrbsInputData)
    assert sample_input_data.input_history[0]["content"] == "test history"
    assert len(sample_input_data.pre_orbs_items) == 1
    assert len(sample_input_data.new_items) == 1
    assert sample_input_data.pre_orbs_items[0].text_content == "pre item"
    assert sample_input_data.new_items[0].text_content == "new item"


def test_orbs_input_filter_with_data(sample_input_data, mock_agent):
    """Test the input filter functionality with actual data."""

    def input_filter(data: OrbsInputData) -> OrbsInputData:
        # Modify the input data
        new_item: ResponseInputItemParam = {
            "type": "message",
            "content": "modified",
            "role": "user",
        }
        return OrbsInputData(
            input_history=data.input_history + (new_item,),
            pre_orbs_items=data.pre_orbs_items,
            new_items=data.new_items,
        )

    orb = Orbs(
        on_invoke_orbs=lambda ctx, input_json=None: None,
        name="test_orbs",
        input_filter=input_filter,
    )

    filtered_data = orb.input_filter(sample_input_data)
    assert filtered_data.input_history[0]["content"] == "test history"
    assert filtered_data.input_history[1]["content"] == "modified"
    assert filtered_data.pre_orbs_items == sample_input_data.pre_orbs_items
    assert filtered_data.new_items == sample_input_data.new_items


def test_orbs_validation_errors(mock_agent):
    """Test validation errors in orbs creation."""

    class InvalidInput:
        """Invalid input class without model_json_schema."""

    async def test_orbs(ctx: RunContextWrapper[Any], input_data: InvalidInput) -> None:
        pass

    # Add input_type to the function
    test_orbs.input_type = InvalidInput

    with pytest.raises(
        UsageError,
        match=ERROR_MESSAGES.ORBS_ERROR.message.format(
            error="type object 'InvalidInput' has no attribute 'model_json_schema'"
        ),
    ):
        # Create orbs instance with invalid input type
        orbs(mock_agent)(test_orbs)


@pytest.mark.asyncio
async def test_orbs_decorator_functionality(
    mock_agent: MockAgent, mock_context: RunContextWrapper[Any]
):
    """Test the orbs decorator with various function types and inputs."""

    # Test with async function and input
    @orbs(mock_agent)
    async def test_orbs_async_with_input(
        ctx: RunContextWrapper[Any], input_data: OrbsTestInput
    ) -> None:
        assert ctx == mock_context
        assert input_data.message == "test message"

    test_orbs_async_with_input.input_type = OrbsTestInput
    result = await test_orbs_async_with_input.on_invoke_orbs(
        mock_context, '{"message": "test message"}'
    )
    assert result == mock_agent

    # Test with async function without input
    @orbs(mock_agent)
    async def test_orbs_async_no_input(ctx: RunContextWrapper[Any]) -> None:
        assert ctx == mock_context

    result = await test_orbs_async_no_input.on_invoke_orbs(mock_context)
    assert result == mock_agent

    # Test with sync function
    @orbs(mock_agent)
    def test_orbs_sync(ctx: RunContextWrapper[Any]) -> None:
        assert ctx == mock_context

    result = await test_orbs_sync.on_invoke_orbs(mock_context)
    assert result == mock_agent

    # Test validation errors
    class InvalidInput(BaseModel):
        """Invalid input class for testing validation errors."""

        required_field: str  # This field is required but won't be provided in the test

    @orbs(mock_agent)
    async def test_orbs_invalid(ctx: RunContextWrapper[Any], input_data: InvalidInput) -> None:
        pass

    test_orbs_invalid.input_type = InvalidInput

    with pytest.raises(
        UsageError,
        match=ERROR_MESSAGES.ORBS_ERROR.message.format(
            error="Invalid input JSON: 1 validation error for InvalidInput"
        ),
    ):
        await test_orbs_invalid.on_invoke_orbs(mock_context, '{"message": "test message"}')

    # Test invalid JSON input
    with pytest.raises(UsageError):
        await test_orbs_async_with_input.on_invoke_orbs(mock_context, '{"invalid": "json"}')
