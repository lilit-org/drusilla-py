"""Tests for the shield module."""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.agent_v1 import AgentV1 as Agent
from src.gear.shield import (
    InputShield,
    ShieldResult,
    input_shield,
    output_shield,
)
from src.util.exceptions import UsageError
from src.util.types import InputItem, RunContextWrapper


class MockAgent(Agent[Any]):
    """Mock agent for testing."""

    def __init__(self, name: str = "test_agent"):
        super().__init__(
            name=name,
            orbs=[],
            swords=[],
            input_shields=[],
            output_shields=[],
        )


@pytest.fixture
def mock_agent() -> MockAgent:
    """Fixture for creating a mock agent."""
    return MockAgent()


@pytest.fixture
def mock_context() -> RunContextWrapper[Any]:
    """Fixture for creating a mock context."""
    return RunContextWrapper(context={})


@pytest.fixture
def input_data() -> str | list[InputItem]:
    """Fixture for creating test input data."""
    return "test input"


@pytest.mark.asyncio
async def test_shield_decorators(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test both input and output shield decorators with sync and async functions."""

    # Test input shield
    @input_shield
    def sync_input_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[InputItem],
    ) -> ShieldResult:
        return ShieldResult(success=True, data="processed")

    @input_shield
    async def async_input_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[InputItem],
    ) -> ShieldResult:
        return ShieldResult(success=True, data="processed")

    # Test output shield
    @output_shield
    def sync_output_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> ShieldResult:
        return ShieldResult(success=True, data="processed")

    @output_shield
    async def async_output_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> ShieldResult:
        return ShieldResult(success=True, data="processed")

    # Run all tests
    for shield in [sync_input_shield, async_input_shield]:
        result = await shield.run(mock_context, mock_agent, "test input")
        assert result.tripwire_triggered is False
        assert result.result == "processed"

    for shield in [sync_output_shield, async_output_shield]:
        result = await shield.run(mock_context, mock_agent, "test output")
        assert result.tripwire_triggered is False
        assert result.result == "processed"


def test_shield_with_name():
    """Test shield with custom name."""

    @input_shield(name="custom_shield")
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[InputItem],
    ) -> ShieldResult:
        return ShieldResult(success=True, data="processed")

    assert test_shield.name == "custom_shield"


@pytest.mark.asyncio
async def test_shield_tripwire(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test shield tripwire functionality."""

    @input_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[InputItem],
    ) -> ShieldResult:
        return ShieldResult(success=False, message="error")

    result = await test_shield.run(mock_context, mock_agent, "test input")
    assert result.tripwire_triggered is True
    assert result.result == "error"


@pytest.mark.asyncio
async def test_shield_error_handling(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test shield error handling."""
    with pytest.raises(UsageError) as exc_info:
        shield = InputShield(shield_function="not a function")
        await shield.run(mock_context, mock_agent, "test input")
    assert str(exc_info.value) == "Shield error: not a function"


@pytest.mark.asyncio
async def test_shield_with_complex_data(
    mock_context: RunContextWrapper[Any], mock_agent: MockAgent
):
    """Test shield with complex input and output data."""

    # Test with list input
    @input_shield
    def test_input_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[InputItem],
    ) -> ShieldResult:
        if isinstance(input_data, list):
            return ShieldResult(success=True, data="processed list")
        return ShieldResult(success=True, data="processed string")

    # Test with complex output
    @output_shield
    def test_output_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> ShieldResult:
        return ShieldResult(
            success=True,
            data={"processed": True, "data": output},
        )

    # Test list input
    input_list = [InputItem(content="item1"), InputItem(content="item2")]
    result = await test_input_shield.run(mock_context, mock_agent, input_list)
    assert result.tripwire_triggered is False
    assert result.result == "processed list"

    # Test string input
    result = await test_input_shield.run(mock_context, mock_agent, "test input")
    assert result.tripwire_triggered is False
    assert result.result == "processed string"

    # Test complex output
    result = await test_output_shield.run(mock_context, mock_agent, {"key": "value"})
    assert result.tripwire_triggered is False
    assert result.result == {"processed": True, "data": {"key": "value"}}
