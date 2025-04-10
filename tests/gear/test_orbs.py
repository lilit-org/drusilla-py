"""Tests for the orbs module."""

from typing import Any

import pytest
from pydantic import BaseModel

from src.agents.agent_v1 import AgentV1 as Agent
from src.gear.orbs import Orbs, OrbsInputData
from src.runners.items import MessageOutputItem
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


def test_orbs_input_data_creation(sample_input_data):
    """Test OrbsInputData creation and properties."""
    assert isinstance(sample_input_data, OrbsInputData)
    assert sample_input_data.input_history[0]["content"] == "test history"
    assert len(sample_input_data.pre_orbs_items) == 1
    assert len(sample_input_data.new_items) == 1
    assert sample_input_data.pre_orbs_items[0].text_content == "pre item"
    assert sample_input_data.new_items[0].text_content == "new item"
