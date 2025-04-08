"""Unit tests for the Agent module."""

from dataclasses import dataclass
from typing import Any

import pytest

from src.agents.agent_v1 import AgentV1 as Agent
from src.agents.agent_v1 import AgentV1OutputSchema
from src.gear.shield import InputShield, ShieldResult
from src.gear.sword import Sword, SwordResult
from src.models.interface import Model
from src.models.settings import ModelSettings
from src.runners.items import ModelResponse, SwordCallOutputItem, Usage
from src.runners.result import RunResult, SwordsToFinalOutputResult
from src.runners.run_impl import RunImpl
from src.util.exceptions import AgentExecutionError, ModelError, UsageError
from src.util.types import RunContextWrapper


@pytest.fixture
def mock_run_context():
    class MockRunContext:
        def __init__(self):
            self.data = {
                "turn": 1,
                "max_turns": 10,
                "input": "test input",
                "output": "test output",
                "metadata": {"test": "value"},
            }

        def get(self, key: str, default: Any = None) -> Any:
            return self.data.get(key, default)

        def set(self, key: str, value: Any) -> None:
            self.data[key] = value

        def update(self, **kwargs: Any) -> None:
            self.data.update(kwargs)

    return RunContextWrapper(MockRunContext())


@pytest.fixture
def mock_sword():
    class MockSword(Sword):
        def __init__(self, name: str = "test_sword", result: str = "sword_result"):
            super().__init__(
                name=name,
                description="Test sword",
                params_json_schema={},
                on_invoke_sword=lambda ctx, input: None,
            )
            self._result = result

        async def __call__(self, ctx: RunContextWrapper[Any], input: str) -> Any:
            return f"{self._result}_{input}"

        def clone(self, **kwargs: Any) -> "MockSword":
            """Create a copy of the sword with optional field updates."""
            new_sword = MockSword(
                name=kwargs.get("name", self.name), result=kwargs.get("result", self._result)
            )
            return new_sword

    return MockSword()


@pytest.fixture
def mock_shield():
    class MockShield(InputShield[Any]):
        def __init__(self, result: str = "shielded"):
            super().__init__(
                shield_function=lambda ctx, agent, input: ShieldResult(
                    tripwire_triggered=False, result=result
                )
            )
            self._result = result

        async def __call__(self, input: str) -> str:
            return f"{self._result}_{input}"

    return MockShield()


@pytest.fixture
def basic_agent():
    return Agent(
        name="test_agent",
        instructions="Test instructions",
        model="test_model",
        model_settings=ModelSettings(),
    )


@pytest.fixture
def mock_model():
    class MockModel(Model):
        def __init__(self):
            self.name = "test_model"
            self.provider = "test_provider"
            self.settings = ModelSettings()

        async def generate(self, prompt: str, **kwargs: Any) -> str:
            return "test response"

        async def get_response(
            self,
            system_instructions: str | None,
            input: list[Any],
            model_settings: Any,
            swords: list[Any],
            output_schema: Any | None,
            orbs: list[Any],
        ) -> ModelResponse:
            # If output_schema is provided, return a response that matches the schema
            if output_schema:
                return ModelResponse(
                    output=[
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": '{"field1": "value1"}'}],
                            "role": "assistant",
                        }
                    ],
                    usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
                    referenceable_id="test_id",
                )
            # Otherwise return a plain text response
            return ModelResponse(
                output=[
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "test response"}],
                        "role": "assistant",
                    }
                ],
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
                referenceable_id="test_id",
            )

        def to_dict(self) -> dict[str, Any]:
            return {
                "name": self.name,
                "provider": self.provider,
                "settings": self.settings.to_dict(),
            }

    return MockModel()


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


@pytest.mark.asyncio
async def test_get_system_prompt_invalid_type(mock_run_context):
    """Test that get_system_prompt raises an error for invalid instruction types."""
    agent = Agent(name="test_agent", instructions=42)  # type: ignore
    with pytest.raises(AgentExecutionError, match="Invalid instructions type: <class 'int'>"):
        await agent.get_system_prompt(mock_run_context)


@pytest.mark.asyncio
async def test_get_system_prompt_none(mock_run_context):
    """Test get_system_prompt when instructions is None."""
    agent = Agent(name="test_agent", instructions=None, model="test_model")
    prompt = await agent.get_system_prompt(mock_run_context)
    assert prompt is None


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


def test_agent_as_sword_custom_description(basic_agent):
    """Test converting agent to sword with custom description."""
    sword = basic_agent.as_sword(sword_description="Custom sword description")
    assert isinstance(sword, Sword)
    assert sword.description == "Custom sword description"


@pytest.mark.asyncio
async def test_agent_with_shields(mock_run_context, basic_agent, mock_shield):
    """Test agent with input and output shields."""
    agent = basic_agent.clone(input_shields=[mock_shield], output_shields=[mock_shield])
    assert len(agent.input_shields) == 1
    assert len(agent.output_shields) == 1

    # Test shield processing
    input_result = await agent.input_shields[0]("test_input")
    assert input_result == "shielded_test_input"

    output_result = await agent.output_shields[0]("test_output")
    assert output_result == "shielded_test_output"


@pytest.mark.asyncio
async def test_agent_with_swords(mock_run_context, basic_agent, mock_sword):
    """Test agent with swords."""
    agent = basic_agent.clone(swords=[mock_sword])
    assert len(agent.swords) == 1
    assert agent.swords[0].name == "test_sword"

    # Test sword invocation
    result = await agent.swords[0](mock_run_context, "test_input")
    assert result == "sword_result_test_input"


def test_swords_to_final_output_result():
    """Test SwordsToFinalOutputResult dataclass."""
    result = SwordsToFinalOutputResult(is_final_output=True, final_output="test_output")
    assert result.is_final_output is True
    assert result.final_output == "test_output"

    # Test with different values
    result2 = SwordsToFinalOutputResult(is_final_output=False, final_output=None)
    assert result2.is_final_output is False
    assert result2.final_output is None


@pytest.mark.asyncio
async def test_agent_with_orbs(mock_run_context, basic_agent):
    """Test agent with orbs configuration."""
    agent = basic_agent.clone(
        orbs_description="Test orbs description", orbs=[basic_agent.clone(name="orb_agent")]
    )
    assert agent.orbs_description == "Test orbs description"
    assert len(agent.orbs) == 1
    assert agent.orbs[0].name == "orb_agent"

    # Test with empty orbs
    agent_empty = basic_agent.clone(orbs=[])
    assert len(agent_empty.orbs) == 0
    assert agent_empty.orbs_description is None


def test_agent_sword_use_behavior(basic_agent):
    """Test different sword use behavior configurations."""
    # Test default behavior
    assert basic_agent.sword_use_behavior == "run_llm_again"

    # Test stop_on_first_sword behavior
    agent_stop = basic_agent.clone(sword_use_behavior="stop_on_first_sword")
    assert agent_stop.sword_use_behavior == "stop_on_first_sword"

    # Test custom sword names behavior
    custom_behavior = {"stop_at_sword_names": ["sword1", "sword2"]}
    agent_custom = basic_agent.clone(sword_use_behavior=custom_behavior)
    assert agent_custom.sword_use_behavior == custom_behavior


@pytest.mark.asyncio
async def test_agent_with_invalid_sword_use_behavior(mock_run_context):
    """Test agent with invalid sword_use_behavior."""
    # Create a mock sword result
    mock_sword = Sword(
        name="test_sword",
        description="Test sword",
        params_json_schema={},
        on_invoke_sword=lambda ctx, input: None,
    )
    sword_result = SwordResult(
        sword=mock_sword,
        output="test output",
        run_item=SwordCallOutputItem(output="test output", raw_item=None, agent=None),
    )

    agent = Agent(
        name="invalid_behavior_agent", sword_use_behavior="invalid_behavior"  # type: ignore
    )

    with pytest.raises(UsageError, match="Invalid sword_use_behavior"):
        await RunImpl._check_for_final_output_from_swords(
            agent=agent, sword_results=[sword_result], context_wrapper=mock_run_context
        )


@pytest.mark.asyncio
async def test_agent_run_basic(mock_run_context, basic_agent, mock_model):
    """Test basic agent run functionality."""
    from src.runners.run import Runner

    agent = basic_agent.clone(
        instructions="You are a test agent. Respond with 'test response'", model=mock_model
    )

    # Run the agent using the Runner class
    result = await Runner.run(
        starting_agent=agent, input="test input", context=mock_run_context, max_turns=1
    )

    assert result is not None
    assert isinstance(result, RunResult)
    assert result.final_output == "test response"
    assert result.is_complete is True
    assert len(result.new_items) > 0
    assert len(result.raw_responses) > 0


@pytest.mark.asyncio
async def test_agent_with_custom_model_settings(mock_run_context):
    """Test agent with custom model settings."""
    custom_settings = ModelSettings(temperature=0.8, max_tokens=1000, top_p=0.9)
    agent = Agent(name="custom_settings_agent", model_settings=custom_settings)

    assert agent.model_settings.temperature == 0.8
    assert agent.model_settings.max_tokens == 1000
    assert agent.model_settings.top_p == 0.9


@pytest.mark.asyncio
async def test_agent_with_output_type(mock_run_context):
    """Test agent with output type specification."""

    @dataclass
    class CustomOutput:
        value: str

    agent = Agent(name="output_type_agent", output_type=CustomOutput)
    assert agent.output_type == CustomOutput


@pytest.mark.asyncio
async def test_agent_with_charms(mock_run_context):
    """Test agent with charms configuration."""
    from src.gear.charms import AgentCharms

    class TestCharms(AgentCharms[Any]):
        async def on_start(self, context: RunContextWrapper[Any], agent: Agent[Any]) -> None:
            context.context.data["charm_started"] = True

        async def on_end(self, context: RunContextWrapper[Any], agent: Agent[Any]) -> None:
            context.context.data["charm_ended"] = True

    charms = TestCharms()
    agent = Agent(name="charms_agent", charms=charms)

    assert agent.charms == charms
    await agent.charms.on_start(mock_run_context, agent)
    assert mock_run_context.context.data["charm_started"] is True
    await agent.charms.on_end(mock_run_context, agent)
    assert mock_run_context.context.data["charm_ended"] is True


@pytest.mark.asyncio
async def test_agent_with_custom_output_extractor(mock_run_context, basic_agent, mock_model):
    """Test agent with custom output extractor when converting to sword."""

    def custom_extractor(output: Any) -> str:
        return f"extracted_{output}"

    # Create a sword with a custom output extractor
    sword = basic_agent.clone(model=mock_model).as_sword(
        sword_name="test_sword",
        sword_description="Test sword with custom extractor",
        custom_output_extractor=custom_extractor,
    )

    # Test that the custom extractor is used in the sword
    result = await sword.on_invoke_sword(mock_run_context, '{"input": "test_input"}')
    assert result.startswith("extracted_")


def test_agent_output_schema_plain_text():
    """Test AgentV1OutputSchema with plain text output."""
    schema = AgentV1OutputSchema(str)
    assert schema.is_plain_text() is True
    assert schema.output_type_name() == "str"
    with pytest.raises(UsageError, match="No JSON schema for plain text output"):
        schema.json_schema()


def test_agent_output_schema_custom_type():
    """Test AgentV1OutputSchema with custom output type."""

    @dataclass
    class CustomOutput:
        value: str
        count: int

    schema = AgentV1OutputSchema(CustomOutput)
    assert schema.is_plain_text() is False
    assert schema.output_type_name() == "CustomOutput"

    # Test JSON schema generation
    json_schema = schema.json_schema()
    assert json_schema["type"] == "object"
    assert "value" in json_schema["properties"]
    assert "count" in json_schema["properties"]

    # Test JSON validation
    valid_json = '{"value": "test", "count": 42}'
    result = schema.validate_json(valid_json)
    assert isinstance(result, CustomOutput)
    assert result.value == "test"
    assert result.count == 42

    # Test invalid JSON
    invalid_json = '{"value": "test", "count": "not a number"}'
    with pytest.raises(ModelError):
        schema.validate_json(invalid_json)


@pytest.mark.asyncio
async def test_agent_with_stop_on_first_sword(mock_run_context, basic_agent, mock_sword):
    """Test agent with stop_on_first_sword behavior."""
    agent = basic_agent.clone(swords=[mock_sword], sword_use_behavior="stop_on_first_sword")
    assert agent.sword_use_behavior == "stop_on_first_sword"

    # Test that the sword is used
    result = await agent.swords[0](mock_run_context, "test_input")
    assert result == "sword_result_test_input"


@pytest.mark.asyncio
async def test_agent_with_stop_at_sword_names(mock_run_context, basic_agent, mock_sword):
    """Test agent with stop_at_sword_names behavior."""
    custom_sword = mock_sword.clone(name="custom_sword")
    agent = basic_agent.clone(
        swords=[mock_sword, custom_sword],
        sword_use_behavior={"stop_at_sword_names": ["custom_sword"]},
    )
    assert isinstance(agent.sword_use_behavior, dict)
    assert agent.sword_use_behavior["stop_at_sword_names"] == ["custom_sword"]

    # Test that both swords are present
    assert len(agent.swords) == 2
    assert agent.swords[0].name == "test_sword"
    assert agent.swords[1].name == "custom_sword"


@pytest.mark.asyncio
async def test_agent_with_async_instructions(mock_run_context):
    """Test agent with async instructions function."""

    async def async_instructions(ctx: RunContextWrapper[Any], agent: Agent[Any]) -> str:
        return f"Async instructions for {agent.name}"

    agent = Agent(name="test_agent", instructions=async_instructions, model="test_model")

    prompt = await agent.get_system_prompt(mock_run_context)
    assert prompt == "Async instructions for test_agent"


@pytest.mark.asyncio
async def test_agent_with_sync_instructions(mock_run_context):
    """Test agent with sync instructions function."""

    def sync_instructions(ctx: RunContextWrapper[Any], agent: Agent[Any]) -> str:
        return f"Sync instructions for {agent.name}"

    agent = Agent(name="test_agent", instructions=sync_instructions, model="test_model")

    prompt = await agent.get_system_prompt(mock_run_context)
    assert prompt == "Sync instructions for test_agent"
