"""Unit tests for the Agent module."""

from typing import Any

import pytest

from src.agents.agent import Agent, SwordsToFinalOutputResult
from src.gear.shield import InputShield, ShieldFunctionOutput
from src.gear.sword import Sword
from src.models.settings import ModelSettings
from src.util._types import RunContextWrapper


# Test fixtures
@pytest.fixture
def mock_run_context():
    class MockRunContext:
        def __init__(self):
            self.data = {}

    return RunContextWrapper(MockRunContext())


@pytest.fixture
def mock_sword():
    class MockSword(Sword):
        def __init__(self, name: str = "test_sword"):
            super().__init__(
                name=name,
                description="Test sword",
                params_json_schema={},
                on_invoke_sword=lambda ctx, input: None,
            )

        async def __call__(self, input: str) -> Any:
            return f"sword_result_{input}"

    return MockSword()


@pytest.fixture
def mock_shield():
    class MockShield(InputShield[Any]):
        def __init__(self):
            super().__init__(
                shield_function=lambda ctx, agent, input: ShieldFunctionOutput(
                    tripwire_triggered=False, output=f"shielded_{input}"
                )
            )

        async def __call__(self, input: str) -> str:
            return f"shielded_{input}"

    return MockShield()


@pytest.fixture
def basic_agent():
    return Agent(
        name="test_agent",
        instructions="Test instructions",
        model="test_model",
        model_settings=ModelSettings(),
    )


# Test cases
def test_agent_initialization(basic_agent):
    """Test basic agent initialization."""
    assert basic_agent.name == "test_agent"
    assert basic_agent.instructions == "Test instructions"
    assert basic_agent.model == "test_model"
    assert isinstance(basic_agent.model_settings, ModelSettings)
    assert basic_agent.swords == []
    assert basic_agent.input_shields == []
    assert basic_agent.output_shields == []
    assert basic_agent.output_type is None
    assert basic_agent.charms is None
    assert basic_agent.sword_use_behavior == "run_llm_again"


def test_agent_clone(basic_agent):
    """Test agent cloning functionality."""
    cloned_agent = basic_agent.clone(name="cloned_agent")
    assert cloned_agent.name == "cloned_agent"
    assert cloned_agent.instructions == basic_agent.instructions
    assert cloned_agent.model == basic_agent.model
    assert cloned_agent.model_settings == basic_agent.model_settings


@pytest.mark.asyncio
async def test_get_system_prompt_string(mock_run_context, basic_agent):
    """Test getting system prompt when instructions is a string."""
    prompt = await basic_agent.get_system_prompt(mock_run_context)
    assert prompt == "Test instructions"


@pytest.mark.asyncio
async def test_get_system_prompt_callable(mock_run_context):
    """Test getting system prompt when instructions is a callable."""

    async def instruction_func(ctx: RunContextWrapper[Any], agent: Agent[Any]) -> str:
        return f"Dynamic instructions for {agent.name}"

    agent = Agent(name="test_agent", instructions=instruction_func, model="test_model")

    prompt = await agent.get_system_prompt(mock_run_context)
    assert prompt == "Dynamic instructions for test_agent"


def test_agent_as_sword(basic_agent):
    """Test converting agent to sword."""
    sword = basic_agent.as_sword()
    assert isinstance(sword, Sword)
    assert sword.name == "test_agent"


def test_agent_as_sword_custom_name(basic_agent):
    """Test converting agent to sword with custom name."""
    sword = basic_agent.as_sword(sword_name="custom_sword")
    assert isinstance(sword, Sword)
    assert sword.name == "custom_sword"


@pytest.mark.asyncio
async def test_agent_with_shields(mock_run_context, basic_agent, mock_shield):
    """Test agent with input and output shields."""
    agent = basic_agent.clone(input_shields=[mock_shield], output_shields=[mock_shield])
    assert len(agent.input_shields) == 1
    assert len(agent.output_shields) == 1


@pytest.mark.asyncio
async def test_agent_with_swords(mock_run_context, basic_agent, mock_sword):
    """Test agent with swords."""
    agent = basic_agent.clone(swords=[mock_sword])
    assert len(agent.swords) == 1
    assert agent.swords[0].name == "test_sword"


def test_swords_to_final_output_result():
    """Test SwordsToFinalOutputResult dataclass."""
    result = SwordsToFinalOutputResult(is_final_output=True, final_output="test_output")
    assert result.is_final_output is True
    assert result.final_output == "test_output"
