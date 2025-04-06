"""Tests for the charm module."""

from typing import Any

import pytest

from src.agents.agent import Agent
from src.gear.charm import AgentCharms, BaseCharms, RunCharms
from src.gear.sword import Sword
from src.util._types import RunContextWrapper


class MockAgent(Agent[Any]):
    """Mock agent for testing."""

    def __init__(self, name: str = "test_agent"):
        self.name = name


@pytest.fixture
def mock_agent() -> MockAgent:
    """Fixture for creating a mock agent."""
    return MockAgent()


@pytest.fixture
def mock_context() -> RunContextWrapper[Any]:
    """Fixture for creating a mock context."""
    return RunContextWrapper(context={})


@pytest.fixture
def mock_sword() -> Sword:
    """Fixture for creating a mock sword."""
    return Sword(
        name="test_sword",
        description="Test sword",
        params_json_schema={},
        on_invoke_sword=lambda ctx, input: None,
    )


class MockBaseCharms(BaseCharms[Any]):
    """Implementation of BaseCharms for testing."""

    def __init__(self):
        self.start_called = False
        self.end_called = False
        self.sword_start_called = False
        self.sword_end_called = False

    async def on_start(self, context: RunContextWrapper[Any], agent: Agent[Any]) -> None:
        self.start_called = True

    async def on_end(self, context: RunContextWrapper[Any], agent: Agent[Any], output: Any) -> None:
        self.end_called = True

    async def on_sword_start(
        self, context: RunContextWrapper[Any], agent: Agent[Any], sword: Sword
    ) -> None:
        self.sword_start_called = True

    async def on_sword_end(
        self, context: RunContextWrapper[Any], agent: Agent[Any], sword: Sword, result: Any
    ) -> None:
        self.sword_end_called = True


class MockRunCharms(RunCharms[Any]):
    """Implementation of RunCharms for testing."""

    def __init__(self):
        super().__init__()
        self.orbs_called = False

    async def on_orbs(
        self, context: RunContextWrapper[Any], from_agent: Agent[Any], to_agent: Agent[Any]
    ) -> None:
        self.orbs_called = True


class MockAgentCharms(AgentCharms[Any]):
    """Implementation of AgentCharms for testing."""

    def __init__(self):
        super().__init__()
        self.orbs_called = False

    async def on_orbs(
        self, context: RunContextWrapper[Any], agent: Agent[Any], source: Agent[Any]
    ) -> None:
        self.orbs_called = True


@pytest.mark.asyncio
async def test_base_charms(
    mock_context: RunContextWrapper[Any], mock_agent: MockAgent, mock_sword: Sword
):
    """Test base charms functionality."""
    charms = MockBaseCharms()

    await charms.on_start(mock_context, mock_agent)
    assert charms.start_called

    await charms.on_end(mock_context, mock_agent, "test output")
    assert charms.end_called

    await charms.on_sword_start(mock_context, mock_agent, mock_sword)
    assert charms.sword_start_called

    await charms.on_sword_end(mock_context, mock_agent, mock_sword, "test result")
    assert charms.sword_end_called


@pytest.mark.asyncio
async def test_run_charms(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test run charms functionality."""
    charms = MockRunCharms()
    other_agent = MockAgent("other_agent")

    await charms.on_orbs(mock_context, mock_agent, other_agent)
    assert charms.orbs_called

    # Verify base charm methods are still callable
    await charms.on_start(mock_context, mock_agent)
    await charms.on_end(mock_context, mock_agent, "test output")


@pytest.mark.asyncio
async def test_agent_charms(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test agent charms functionality."""
    charms = MockAgentCharms()
    source_agent = MockAgent("source_agent")

    await charms.on_orbs(mock_context, mock_agent, source_agent)
    assert charms.orbs_called

    # Verify base charm methods are still callable
    await charms.on_start(mock_context, mock_agent)
    await charms.on_end(mock_context, mock_agent, "test output")


@pytest.mark.asyncio
async def test_charm_protocol_compliance():
    """Test that charm implementations comply with the protocol."""
    # This test verifies that our implementations have all required methods
    # with correct signatures
    base_charms = MockBaseCharms()
    run_charms = MockRunCharms()
    agent_charms = MockAgentCharms()

    # If we can create instances and they have all required methods,
    # they comply with the protocol
    assert hasattr(base_charms, "on_start")
    assert hasattr(base_charms, "on_end")
    assert hasattr(base_charms, "on_sword_start")
    assert hasattr(base_charms, "on_sword_end")

    assert hasattr(run_charms, "on_orbs")
    assert hasattr(agent_charms, "on_orbs")
