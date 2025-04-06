"""Tests for the shield module."""

from typing import Any

import pytest

from src.agents.agent import Agent
from src.gear.shield import (
    InputShield,
    InputShieldResult,
    OutputShieldResult,
    ShieldFunctionOutput,
    input_shield,
    output_shield,
)
from src.util._items import TResponseInputItem
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


@pytest.mark.asyncio
async def test_input_shield_sync(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test synchronous input shield."""

    @input_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[TResponseInputItem],
    ) -> ShieldFunctionOutput:
        return ShieldFunctionOutput(tripwire_triggered=False, output="processed")

    result = await test_shield.run(mock_context, mock_agent, "test input")
    assert isinstance(result, InputShieldResult)
    assert result.shield == test_shield
    assert result.agent == mock_agent
    assert result.input == "test input"
    assert result.output.tripwire_triggered is False
    assert result.output.output == "processed"


@pytest.mark.asyncio
async def test_input_shield_async(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test asynchronous input shield."""

    @input_shield
    async def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[TResponseInputItem],
    ) -> ShieldFunctionOutput:
        return ShieldFunctionOutput(tripwire_triggered=False, output="processed")

    result = await test_shield.run(mock_context, mock_agent, "test input")
    assert isinstance(result, InputShieldResult)
    assert result.output.tripwire_triggered is False
    assert result.output.output == "processed"


@pytest.mark.asyncio
async def test_output_shield_sync(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test synchronous output shield."""

    @output_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> ShieldFunctionOutput:
        return ShieldFunctionOutput(tripwire_triggered=False, output="processed")

    result = await test_shield.run(mock_context, mock_agent, "test output")
    assert isinstance(result, OutputShieldResult)
    assert result.shield == test_shield
    assert result.agent == mock_agent
    assert result.agent_output == "test output"
    assert result.output.tripwire_triggered is False
    assert result.output.output == "processed"


@pytest.mark.asyncio
async def test_output_shield_async(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test asynchronous output shield."""

    @output_shield
    async def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> ShieldFunctionOutput:
        return ShieldFunctionOutput(tripwire_triggered=False, output="processed")

    result = await test_shield.run(mock_context, mock_agent, "test output")
    assert isinstance(result, OutputShieldResult)
    assert result.output.tripwire_triggered is False
    assert result.output.output == "processed"


def test_shield_with_name():
    """Test shield with custom name."""

    @input_shield(name="custom_shield")
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[TResponseInputItem],
    ) -> ShieldFunctionOutput:
        return ShieldFunctionOutput(tripwire_triggered=False)

    assert test_shield.name == "custom_shield"


@pytest.mark.asyncio
async def test_shield_tripwire(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test shield tripwire functionality."""

    @input_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[TResponseInputItem],
    ) -> ShieldFunctionOutput:
        return ShieldFunctionOutput(tripwire_triggered=True, output="error")

    result = await test_shield.run(mock_context, mock_agent, "test input")
    assert result.output.tripwire_triggered is True
    assert result.output.output == "error"


@pytest.mark.asyncio
async def test_shield_error_handling(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test shield error handling."""
    with pytest.raises(TypeError, match="shield_function must be callable"):
        shield = InputShield(shield_function="not a function")
        await shield.run(mock_context, mock_agent, "test input")


@pytest.mark.asyncio
async def test_shield_with_list_input(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test shield with list input."""

    @input_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[TResponseInputItem],
    ) -> ShieldFunctionOutput:
        assert isinstance(input_data, list)
        return ShieldFunctionOutput(tripwire_triggered=False, output="processed")

    input_list = [TResponseInputItem(content="item1"), TResponseInputItem(content="item2")]
    result = await test_shield.run(mock_context, mock_agent, input_list)
    assert result.output.tripwire_triggered is False
    assert result.output.output == "processed"


@pytest.mark.asyncio
async def test_shield_with_complex_output(
    mock_context: RunContextWrapper[Any], mock_agent: MockAgent
):
    """Test shield with complex output."""

    @output_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> ShieldFunctionOutput:
        return ShieldFunctionOutput(
            tripwire_triggered=False, output={"processed": True, "data": output}
        )

    result = await test_shield.run(mock_context, mock_agent, {"key": "value"})
    assert result.output.tripwire_triggered is False
    assert result.output.output == {"processed": True, "data": {"key": "value"}}


@pytest.mark.asyncio
async def test_shield_with_none_output(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test shield with None output."""

    @output_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> ShieldFunctionOutput:
        return ShieldFunctionOutput(tripwire_triggered=False, output=None)

    result = await test_shield.run(mock_context, mock_agent, "test output")
    assert result.output.tripwire_triggered is False
    assert result.output.output is None
