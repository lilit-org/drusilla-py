"""Tests for the shield module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.agents.agent import Agent
from src.gear.shield import (
    InputShield,
    ShieldResult,
    input_shield,
    output_shield,
)
from src.util._constants import ERROR_MESSAGES
from src.util._exceptions import UsageError
from src.util._types import InputItem, RunContextWrapper

if TYPE_CHECKING:
    from src.agents.agent import Agent


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
async def test_input_shield_sync(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test synchronous input shield."""

    @input_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[InputItem],
    ) -> ShieldResult:
        return ShieldResult(success=True, data="processed")

    result = await test_shield.run(mock_context, mock_agent, "test input")
    assert result.tripwire_triggered is False
    assert result.result == "processed"


@pytest.mark.asyncio
async def test_input_shield_async(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test asynchronous input shield."""

    @input_shield
    async def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[InputItem],
    ) -> ShieldResult:
        return ShieldResult(success=True, data="processed")

    result = await test_shield.run(mock_context, mock_agent, "test input")
    assert result.tripwire_triggered is False
    assert result.result == "processed"


@pytest.mark.asyncio
async def test_output_shield_sync(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test synchronous output shield."""

    @output_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> ShieldResult:
        return ShieldResult(success=True, data="processed")

    result = await test_shield.run(mock_context, mock_agent, "test output")
    assert result.tripwire_triggered is False
    assert result.result == "processed"


@pytest.mark.asyncio
async def test_output_shield_async(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test asynchronous output shield."""

    @output_shield
    async def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> ShieldResult:
        return ShieldResult(success=True, data="processed")

    result = await test_shield.run(mock_context, mock_agent, "test output")
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
    with pytest.raises(
        UsageError,
        match=ERROR_MESSAGES.SHIELD_ERROR.message.format(error="not a function"),
    ):
        shield = InputShield(shield_function="not a function")
        await shield.run(mock_context, mock_agent, "test input")


@pytest.mark.asyncio
async def test_shield_with_list_input(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test shield with list input."""

    @input_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        input_data: str | list[InputItem],
    ) -> ShieldResult:
        assert isinstance(input_data, list)
        return ShieldResult(success=True, data="processed")

    input_list = [InputItem(content="item1"), InputItem(content="item2")]
    result = await test_shield.run(mock_context, mock_agent, input_list)
    assert result.tripwire_triggered is False
    assert result.result == "processed"


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
    ) -> ShieldResult:
        return ShieldResult(
            success=True,
            data={"processed": True, "data": output},
        )

    result = await test_shield.run(mock_context, mock_agent, {"key": "value"})
    assert result.tripwire_triggered is False
    assert result.result == {"processed": True, "data": {"key": "value"}}


@pytest.mark.asyncio
async def test_shield_with_none_output(mock_context: RunContextWrapper[Any], mock_agent: MockAgent):
    """Test shield with None output."""

    @output_shield
    def test_shield(
        ctx: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> ShieldResult:
        return ShieldResult(success=True, data=None)

    result = await test_shield.run(mock_context, mock_agent, "test output")
    assert result.tripwire_triggered is False
    assert result.result is None


@pytest.mark.asyncio
async def test_input_shield_with_list_input(
    mock_context: RunContextWrapper[Any],
    mock_agent: Agent[Any],
    input_data: str | list[InputItem],
):
    shield = InputShield(lambda ctx, agent, data: ShieldResult(success=True, data="processed"))
    result = await shield.run(mock_context, mock_agent, input_data)
    assert result.tripwire_triggered is False
    assert result.result == "processed"


@pytest.mark.asyncio
async def test_input_shield_with_string_input(
    mock_context: RunContextWrapper[Any],
    mock_agent: Agent[Any],
    input_data: str | list[InputItem],
):
    shield = InputShield(lambda ctx, agent, data: ShieldResult(success=True, data="processed"))
    result = await shield.run(mock_context, mock_agent, input_data)
    assert result.tripwire_triggered is False
    assert result.result == "processed"


@pytest.mark.asyncio
async def test_input_shield_with_custom_validation(
    mock_context: RunContextWrapper[Any],
    mock_agent: Agent[Any],
    input_data: str | list[InputItem],
):
    shield = InputShield(lambda ctx, agent, data: ShieldResult(success=True, data="processed"))
    result = await shield.run(mock_context, mock_agent, input_data)
    assert result.tripwire_triggered is False
    assert result.result == "processed"


@pytest.mark.asyncio
async def test_input_shield_with_async_validation(
    mock_context: RunContextWrapper[Any],
    mock_agent: Agent[Any],
    input_data: str | list[InputItem],
):
    shield = InputShield(lambda ctx, agent, data: ShieldResult(success=True, data="processed"))
    result = await shield.run(mock_context, mock_agent, input_data)
    assert result.tripwire_triggered is False
    assert result.result == "processed"


@pytest.mark.asyncio
async def test_input_shield_with_tripwire(
    mock_context: RunContextWrapper[Any],
    mock_agent: Agent[Any],
    input_data: str | list[InputItem],
):
    shield = InputShield(lambda ctx, agent, data: ShieldResult(success=True, data="processed"))
    result = await shield.run(mock_context, mock_agent, input_data)
    assert result.tripwire_triggered is False
    assert result.result == "processed"


@pytest.mark.asyncio
async def test_input_shield_with_list_input_and_tripwire(
    mock_context: RunContextWrapper[Any],
    mock_agent: Agent[Any],
):
    input_list = [InputItem(content="item1"), InputItem(content="item2")]
    shield = InputShield(lambda ctx, agent, data: ShieldResult(success=True, data="processed"))
    result = await shield.run(mock_context, mock_agent, input_list)
    assert result.tripwire_triggered is False
    assert result.result == "processed"
