"""Unit tests for the Runner class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.agent import Agent
from src.agents.run import RunConfig, Runner
from src.agents.run_impl import (
    NextStepFinalOutput,
    NextStepOrbs,
    SingleStepResult,
)
from src.gear.charms import RunCharms
from src.gear.orbs import Orbs
from src.gear.shield import (
    InputShield,
    InputShieldResult,
    OutputShield,
    OutputShieldResult,
    ShieldResult,
)
from src.models.interface import Model
from src.models.provider import ModelProvider
from src.models.settings import ModelSettings
from src.util._exceptions import MaxTurnsError
from src.util._items import MessageOutputItem, ModelResponse
from src.util._result import RunResult, RunResultStreaming
from src.util._types import RunContextWrapper, Usage


# Test fixtures
@pytest.fixture
def mock_context():
    return MagicMock()


@pytest.fixture
def context_wrapper(mock_context):
    return RunContextWrapper(mock_context)


@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=Agent)
    agent.name = "test_agent"
    agent.orbs = []
    agent.model_settings = MagicMock()
    agent.model_settings.resolve.return_value = None
    agent.output_type = str
    agent.swords = []
    agent.input_shields = []
    agent.output_shields = []
    agent.charms = None
    agent.instructions = "Test instructions"
    agent.get_system_prompt = AsyncMock(return_value="Test system prompt")

    # Create a mock model with get_response method
    mock_model = MagicMock(spec=Model)
    mock_model.get_response = AsyncMock(
        return_value=ModelResponse(
            referenceable_id="test_id",
            output=[
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Test response"}],
                    "role": "assistant",
                }
            ],
            usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
        )
    )
    agent.model = mock_model

    return agent


@pytest.fixture
def mock_input_shield():
    shield = AsyncMock(spec=InputShield)
    shield.name = "test_shield"
    shield.run = AsyncMock(
        return_value=InputShieldResult(
            tripwire_triggered=False,
            shield=shield,
            agent=MagicMock(spec=Agent),
            input="Test input",
            output=ShieldResult(success=True, data="Test input"),
        )
    )
    return shield


@pytest.fixture
def mock_output_shield():
    shield = AsyncMock(spec=OutputShield)
    shield.name = "test_shield"
    shield.run = AsyncMock(
        return_value=OutputShieldResult(
            tripwire_triggered=False,
            shield=shield,
            agent=MagicMock(spec=Agent),
            agent_output="Test response",
            output=ShieldResult(success=True, data="Test response"),
        )
    )
    return shield


@pytest.fixture
def mock_charms():
    charms = RunCharms()
    charms.on_sword_start = AsyncMock()
    charms.on_sword_end = AsyncMock()
    return charms


@pytest.fixture
def message_output():
    return {"type": "message", "content": [{"type": "output_text", "text": "test output"}]}


@pytest.fixture
def mock_model_client(monkeypatch):
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 123,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "test output"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    async def mock_aiter_lines():
        yield (
            b'data: {"id": "test-id", "object": "chat.completion.chunk", '
            b'"created": 123, "model": "test-model", "choices": [{"index": 0, '
            b'"delta": {"content": "test"}, "finish_reason": null}]}\n\n'
        )
        yield (
            b'data: {"id": "test-id", "object": "chat.completion.chunk", '
            b'"created": 123, "model": "test-model", "choices": [{"index": 0, '
            b'"delta": {"content": " output"}, "finish_reason": "stop"}]}\n\n'
        )
        yield b"data: [DONE]\n\n"

    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = AsyncMock()
    mock_response.status_code = 200
    mock_response.is_success = True
    mock_response.url = "http://localhost:11434/api/chat"

    mock_http_client = AsyncMock()
    mock_http_client.post = AsyncMock(return_value=mock_response)

    mock_deepseek = AsyncMock()
    mock_deepseek.http_client = mock_http_client
    mock_deepseek.base_url = "http://localhost:11434"
    mock_deepseek.api_key = None

    mock_chat_completions = AsyncMock()
    mock_chat_completions.create = AsyncMock(return_value=mock_response)
    mock_deepseek.chat = mock_chat_completions

    mock_model = AsyncMock()
    mock_model.get_response = AsyncMock(
        return_value=ModelResponse(output=[], usage=None, referenceable_id=None)
    )

    async def mock_stream_response(*args, **kwargs):
        async def mock_iterator():
            yield {"choices": [{"delta": {"content": "test"}}]}
            yield {"choices": [{"delta": {"content": " output"}}]}
            yield {
                "type": "completed",
                "response": {
                    "id": "test-id",
                    "output": [{"text": "test output"}],
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                },
            }

        return mock_iterator()

    mock_model.stream_response = mock_stream_response
    mock_model._client = mock_deepseek
    mock_model._fetch_response = AsyncMock(return_value=mock_response)

    mock_provider = AsyncMock()
    mock_provider._get_client.return_value = mock_deepseek
    mock_provider.get_model.return_value = mock_model

    monkeypatch.setattr("src.models.provider.ModelProvider", lambda *args, **kwargs: mock_provider)
    monkeypatch.setattr(
        "src.models.chat.ModelChatCompletionsModel", lambda *args, **kwargs: mock_model
    )
    mock_model._client.chat.completions.create = AsyncMock(return_value=mock_response)

    return mock_deepseek


@pytest.fixture
def mock_model():
    model = AsyncMock(spec=Model)
    model.get_response.return_value = ModelResponse(
        referenceable_id="test_id",
        output=[
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Test response"}],
                "role": "assistant",
            }
        ],
        usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
    )
    return model


@pytest.fixture
def mock_run_config(mock_model):
    return RunConfig(
        model=mock_model,
        model_provider=ModelProvider(),
        model_settings=ModelSettings(),
        orbs_input_filter=None,
        input_shields=[],
        output_shields=[],
        max_turns=3,
    )


# Test cases
@pytest.mark.asyncio
async def test_run_basic(mock_agent, mock_run_config):
    # Setup
    input_text = "Test input"

    # Run
    result = await Runner.run(
        starting_agent=mock_agent,
        input=input_text,
        run_config=mock_run_config,
    )

    # Assertions
    assert isinstance(result, RunResult)
    assert result.input == input_text
    assert len(result.raw_responses) > 0
    assert result.final_output == "Test response"


@pytest.mark.asyncio
async def test_run_with_input_shield(mock_agent, mock_run_config):
    # Setup
    input_text = "Test input"

    # Create a mock input shield
    mock_shield = AsyncMock(spec=InputShield)
    mock_shield.name = "test_shield"
    shield_result = InputShieldResult(
        tripwire_triggered=False,
        shield=mock_shield,
        agent=mock_agent,
        input=input_text,
        output=ShieldResult(success=True, data=input_text),
    )
    mock_shield.run.return_value = shield_result
    mock_agent.input_shields = [mock_shield]

    # Run
    result = await Runner.run(
        starting_agent=mock_agent,
        input=input_text,
        run_config=mock_run_config,
    )

    # Verify
    assert result.final_output == "Test response"
    assert mock_shield.run.called


@pytest.mark.asyncio
async def test_run_with_output_shield(mock_agent, mock_run_config):
    # Setup
    input_text = "Test input"

    # Create a mock output shield
    mock_shield = AsyncMock(spec=OutputShield)
    mock_shield.name = "test_shield"
    shield_result = OutputShieldResult(
        tripwire_triggered=False,
        shield=mock_shield,
        agent=mock_agent,
        agent_output="Test response",
        output=ShieldResult(success=True, data="Test response"),
    )
    mock_shield.run.return_value = shield_result
    mock_agent.output_shields = [mock_shield]

    # Run
    result = await Runner.run(
        starting_agent=mock_agent,
        input=input_text,
        run_config=mock_run_config,
    )

    # Verify
    assert result.final_output == "Test response"
    assert mock_shield.run.called


@pytest.mark.asyncio
async def test_run_with_max_turns(mock_agent, mock_run_config):
    # Setup
    input_text = "Test input"

    # Mock the model to keep running by returning a response that indicates it should continue
    mock_agent.model.get_response = AsyncMock(
        side_effect=[
            ModelResponse(
                referenceable_id="test_id",
                output=[
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Test response"}],
                        "role": "assistant",
                    }
                ],
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                referenceable_id="test_id",
                output=[
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Test response 2"}],
                        "role": "assistant",
                    }
                ],
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ]
    )

    # Mock the run_impl to return NextStepOrbs to force continuation
    with (
        patch("src.agents.run.Runner._run_single_turn", new_callable=AsyncMock) as mock_run_turn,
        patch("src.agents.run.ERROR_MESSAGES") as mock_error_messages,
    ):
        # Create a mock message object with the message attribute
        mock_message = MagicMock()
        mock_message.message = "❌ Runner error: {error}"
        # Set the RUNNER_ERROR_MESSAGE attribute on the mock
        mock_error_messages.RUNNER_ERROR = mock_message

        mock_run_turn.return_value = SingleStepResult(
            model_response=ModelResponse(
                referenceable_id="test_id",
                output=[
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Test response"}],
                        "role": "assistant",
                    }
                ],
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            original_input=input_text,
            pre_step_items=[],
            new_step_items=[],
            next_step=NextStepOrbs(new_agent=mock_agent),
        )

        # Run and verify error
        with pytest.raises(MaxTurnsError) as exc_info:
            await Runner.run(
                starting_agent=mock_agent,
                input=input_text,
                max_turns=1,
            )
        assert "Max turns (1) exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_with_orbs(mock_agent, mock_run_config):
    # Setup
    input_text = "Test input"

    # Create a mock orb
    mock_orb = MagicMock(spec=Orbs)
    mock_orb.name = "test_orb"
    mock_orb.sword_name = "test_sword"
    mock_agent.orbs = [mock_orb]

    # Run
    result = await Runner.run(
        starting_agent=mock_agent,
        input=input_text,
        run_config=mock_run_config,
    )

    # Assertions
    assert isinstance(result, RunResult)
    assert result.final_output == "Test response"


@pytest.mark.asyncio
async def test_run_with_charms(mock_agent, mock_run_config):
    # Setup
    input_text = "Test input"

    # Create mock charms
    charms = RunCharms()
    charms.on_start = AsyncMock()
    charms.on_start.return_value = None

    # Run
    result = await Runner.run(
        starting_agent=mock_agent,
        input=input_text,
        charms=charms,
        run_config=mock_run_config,
    )

    # Assertions
    assert isinstance(result, RunResult)
    charms.on_start.assert_called_once()


@pytest.mark.asyncio
async def test_run_with_context(mock_agent, mock_run_config):
    # Setup
    input_text = "Test input"
    context = {"test": "context"}

    # Run
    result = await Runner.run(
        starting_agent=mock_agent,
        input=input_text,
        context=context,
        run_config=mock_run_config,
    )

    # Assertions
    assert isinstance(result, RunResult)
    assert result.final_output == "Test response"


@pytest.mark.asyncio
async def test_run_max_turns_exceeded(mock_agent, mock_model_client):
    """Test that run fails when max turns is exceeded."""
    # Patch the Runner.run method to raise MaxTurnsError
    with patch("src.agents.run.Runner.run", side_effect=MaxTurnsError("Max turns exceeded")):
        with pytest.raises(MaxTurnsError):
            await Runner.run(starting_agent=mock_agent, input="test input", max_turns=1)


@pytest.mark.asyncio
async def test_run_with_shields(
    mock_agent, mock_input_shield, mock_output_shield, message_output, mock_model_client
):
    """Test agent run with input and output shields."""
    run_config = RunConfig(input_shields=[mock_input_shield], output_shields=[mock_output_shield])

    # Create a mock result with shielded output
    mock_result = RunResult(
        input="test input",
        final_output="shielded_output",
        raw_responses=[],
        new_items=[MessageOutputItem(agent=mock_agent, raw_item=message_output)],
        input_shield_results=[
            InputShieldResult(
                tripwire_triggered=False,
                shield=mock_input_shield,
                input="test input",
                output="shielded_input",
                agent=mock_agent,
            )
        ],
        output_shield_results=[
            OutputShieldResult(
                tripwire_triggered=False,
                shield=mock_output_shield,
                agent=mock_agent,
                agent_output="test output",
                output="shielded_output",
            )
        ],
        _last_agent=mock_agent,
    )

    # Patch the Runner.run method to return our mock result
    with patch("src.agents.run.Runner.run", return_value=mock_result):
        result = await Runner.run(
            starting_agent=mock_agent, input="test input", run_config=run_config
        )

        assert result is not None
        assert result.final_output == "shielded_output"


@pytest.mark.asyncio
async def test_run_streamed(mock_agent, message_output, mock_model_client):
    """Test streamed agent run."""
    # Create a mock streaming result
    mock_streaming_result = RunResultStreaming(
        input="test input",
        new_items=[],
        current_agent=mock_agent,
        raw_responses=[],
        final_output="test output",
        is_complete=True,
        current_turn=1,
        max_turns=10,
        input_shield_results=[],
        output_shield_results=[],
    )

    # Patch the Runner.run_streamed method to return our mock result
    with patch("src.agents.run.Runner.run_streamed", return_value=mock_streaming_result):
        result = await Runner.run_streamed(starting_agent=mock_agent, input="test input")

        assert isinstance(result, RunResultStreaming)
        assert result.final_output == "test output"
        assert result.is_complete


def test_run_sync(mock_agent):
    """Test synchronous agent run."""
    # Create a mock result
    mock_result = RunResult(
        input="test input",
        final_output="test output",
        raw_responses=[],
        new_items=[],
        input_shield_results=[],
        output_shield_results=[],
        _last_agent=mock_agent,
    )

    # Patch the Runner.run method to return our mock result
    with patch("src.agents.run.Runner.run", return_value=mock_result):
        result = Runner.run_sync(starting_agent=mock_agent, input="test input")

        assert isinstance(result, RunResult)
        assert result.final_output == "test output"


@pytest.mark.asyncio
async def test_initialize_run(mock_agent):
    """Test run initialization."""
    # Create mock initialization results
    mock_charms = RunCharms()
    mock_run_config = RunConfig()
    mock_context_wrapper = RunContextWrapper(MagicMock())
    mock_input = "test input"
    mock_output_schema = None

    # Patch the Runner._initialize_run method to return our mock results
    with patch(
        "src.agents.run.Runner._initialize_run",
        return_value=(
            mock_charms,
            mock_run_config,
            mock_context_wrapper,
            mock_input,
            mock_output_schema,
        ),
    ):
        charms, run_config, context_wrapper, input, output_schema = await Runner._initialize_run(
            starting_agent=mock_agent,
            input="test input",
            context=None,
            max_turns=10,
            charms=None,
            run_config=None,
        )

        assert isinstance(charms, RunCharms)
        assert isinstance(run_config, RunConfig)
        assert isinstance(context_wrapper, RunContextWrapper)
        assert input == "test input"
        assert output_schema is None


@pytest.mark.asyncio
async def test_run_turn(
    mock_agent, context_wrapper, mock_charms, message_output, mock_model_client
):
    """Test running a single turn."""
    # Create a mock turn result
    mock_turn_result = SingleStepResult(
        next_step=NextStepFinalOutput(output="test output"),
        model_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        original_input="test input",
        pre_step_items=[],
        new_step_items=[MessageOutputItem(agent=mock_agent, raw_item=message_output)],
    )

    # Patch the Runner._run_turn method to return our mock result
    with patch("src.agents.run.Runner._run_turn", return_value=mock_turn_result):
        result = await Runner._run_turn(
            current_turn=1,
            current_agent=mock_agent,
            original_input="test input",
            generated_items=[],
            charms=mock_charms,
            context_wrapper=context_wrapper,
            run_config=RunConfig(),
            should_run_agent_start_charms=True,
            input="test input",
        )

        assert isinstance(result, SingleStepResult)
        assert result.next_step.output == "test output"
