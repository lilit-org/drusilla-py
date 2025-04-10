"""Unit tests for the Runner class."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.agent_v1 import AgentV1 as Agent
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
from src.runners.items import MessageOutputItem, ModelResponse
from src.runners.result import RunResult, RunResultStreaming
from src.runners.run import RunConfig, RunContextWrapper, Runner
from src.runners.run_impl import (
    NextStepFinalOutput,
    NextStepOrbs,
    SingleStepResult,
)
from src.util.exceptions import MaxTurnsError, RunnerError
from src.util.types import ResponseEvent, Usage


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
    agent.model_settings = MagicMock(spec=ModelSettings)
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
    return {
        "type": "message",
        "content": [{"type": "output_text", "text": "test output"}],
    }


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
            yield ResponseEvent(
                type="output_text.delta",
                delta="test",
            )
            yield ResponseEvent(
                type="output_text.delta",
                delta=" output",
            )
            yield ResponseEvent(
                type="completed",
                response=ModelResponse(
                    referenceable_id="test-id",
                    output=[
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "test output"}],
                        }
                    ],
                    usage=Usage(requests=1, input_tokens=10, output_tokens=20, total_tokens=30),
                ),
            )

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
    with patch("src.runners.run.Runner._run_turn", new_callable=AsyncMock) as mock_run_turn:
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
        with pytest.raises(RunnerError) as exc_info:
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
    with patch("src.runners.run.Runner.run", side_effect=MaxTurnsError("Max turns exceeded")):
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
    with patch("src.runners.run.Runner.run", return_value=mock_result):
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
    with patch("src.runners.run.Runner.run_streamed", return_value=mock_streaming_result):
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

    # Create a new event loop for the test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Patch the Runner.run method to return our mock result
        with patch("src.runners.run.Runner.run", return_value=mock_result):
            result = Runner.run_sync(starting_agent=mock_agent, input="test input")
            assert result == mock_result
    finally:
        # Clean up the event loop
        loop.close()
        asyncio.set_event_loop(None)


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
        "src.runners.run.Runner._initialize_run",
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
    with patch("src.runners.run.Runner._run_turn", return_value=mock_turn_result):
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


@pytest.mark.asyncio
async def test_run_with_input_shield_error(mock_agent):
    # Setup
    input_text = "Test input"
    mock_context = MagicMock(spec=RunContextWrapper)
    mock_context.usage = MagicMock()
    mock_context.context = MagicMock()  # Add the context attribute

    # Create a mock shield that will trigger an error
    shield = AsyncMock(spec=InputShield)
    shield.name = "test_shield"
    shield_result = InputShieldResult(
        tripwire_triggered=True,
        shield=shield,
        agent=mock_agent,
        input=input_text,
        output=ShieldResult(success=False, data="Invalid input", tripwire_triggered=True),
    )
    shield.run = AsyncMock(return_value=shield_result)

    # Create a run config with the shield
    run_config = RunConfig(
        model=mock_agent.model,
        model_provider=ModelProvider(),
        model_settings=ModelSettings(),
        orbs_input_filter=None,
        input_shields=[shield],
        output_shields=[],
        max_turns=3,
    )

    # Mock the context creation and error handling
    with (
        patch("src.runners.run.RunContextWrapper", return_value=mock_context),
        patch(
            "src.runners.run.Runner._run_input_shields", return_value=[shield_result]
        ) as mock_run_shields,
    ):
        # Run and verify error
        with pytest.raises(RunnerError) as exc_info:
            await Runner.run(
                starting_agent=mock_agent,
                input=input_text,
                run_config=run_config,
            )
        assert "Input shield test_shield triggered" in str(exc_info.value)
        mock_run_shields.assert_called_once_with(
            context=mock_context,
            agent=mock_agent,
            input=input_text,
            shields=[shield],
        )


@pytest.mark.asyncio
async def test_run_with_output_shield_error(mock_agent):
    # Setup
    input_text = "Test input"
    mock_context = MagicMock(spec=RunContextWrapper)
    mock_context.usage = MagicMock()

    # Create a mock shield that will trigger an error
    shield = AsyncMock(spec=OutputShield)
    shield.name = "test_shield"
    shield_result = OutputShieldResult(
        tripwire_triggered=True,
        shield=shield,
        agent=mock_agent,
        agent_output="Test response",
        output=ShieldResult(success=False, data="Invalid output", tripwire_triggered=True),
    )
    shield.run = AsyncMock(return_value=shield_result)

    # Create a run config with the shield
    run_config = RunConfig(
        model=mock_agent.model,
        model_provider=ModelProvider(),
        model_settings=ModelSettings(),
        orbs_input_filter=None,
        input_shields=[],
        output_shields=[shield],
        max_turns=3,
    )

    # Mock the context creation and error handling
    with (
        patch("src.runners.run.RunContextWrapper", return_value=mock_context),
        patch(
            "src.runners.run.Runner._run_output_shields", new_callable=AsyncMock
        ) as mock_run_shields,
    ):
        # Set up the mock to return our shield result
        mock_run_shields.return_value = [shield_result]

        # Run and verify error
        with pytest.raises(RunnerError) as exc_info:
            await Runner.run(
                starting_agent=mock_agent,
                input=input_text,
                run_config=run_config,
            )
        assert "Output shield test_shield triggered" in str(exc_info.value)

        # Verify the mock was called with the correct arguments
        mock_run_shields.assert_called_once()
        call_args = mock_run_shields.call_args
        assert call_args.args == (mock_agent, "Test response", mock_context, [shield])


@pytest.mark.asyncio
async def test_run_with_model_error(mock_agent, mock_model):
    # Setup
    input_text = "Test input"
    mock_model.get_response.side_effect = Exception("Model error")
    run_config = RunConfig(model=mock_model)

    # Run and verify error
    with pytest.raises(RunnerError) as exc_info:
        await Runner.run(
            starting_agent=mock_agent,
            input=input_text,
            run_config=run_config,
        )
    assert "Model error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_streamed_with_error(mock_agent, mock_model_client):
    # Setup
    input_text = "Test input"
    mock_model_client.get_response.side_effect = Exception("Streaming error")
    run_config = RunConfig(model=mock_model_client)

    # Run and verify error handling
    result = await Runner.run_streamed(
        starting_agent=mock_agent,
        input=input_text,
        run_config=run_config,
    )
    assert result.final_output is None
    assert result.raw_responses == []


@pytest.mark.asyncio
async def test_run_with_empty_input(mock_agent, mock_run_config):
    # Setup
    input_text = ""

    # Run and verify
    result = await Runner.run(
        starting_agent=mock_agent,
        input=input_text,
        run_config=mock_run_config,
    )
    assert result is not None
    assert result.final_output is not None
    assert len(result.raw_responses) > 0


@pytest.mark.asyncio
async def test_run_with_none_context(mock_agent, mock_run_config):
    # Setup
    input_text = "Test input"

    # Run with None context and verify
    result = await Runner.run(
        starting_agent=mock_agent,
        input=input_text,
        context=None,
        run_config=mock_run_config,
    )
    assert result is not None
    assert result.final_output is not None
    assert len(result.raw_responses) > 0


@pytest.mark.asyncio
async def test_run_with_invalid_next_step(mock_agent, mock_model):
    # Setup
    input_text = "Test input"
    mock_model.get_response.return_value = ModelResponse(
        referenceable_id="test_id",
        output=[
            {
                "type": "invalid_type",
                "content": [{"type": "output_text", "text": "Test response"}],
                "role": "assistant",
            }
        ],
        usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
    )
    run_config = RunConfig(model=mock_model)

    # Mock the run_impl to return an invalid next step
    with patch("src.runners.run.Runner._run_turn", new_callable=AsyncMock) as mock_run_turn:
        mock_run_turn.return_value = SingleStepResult(
            model_response=ModelResponse(
                referenceable_id="test_id",
                output=[],
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            original_input=input_text,
            pre_step_items=[],
            new_step_items=[],
            next_step=MagicMock(),  # Invalid next step type
        )

        # Run and verify error
        with pytest.raises(RunnerError) as exc_info:
            await Runner.run(
                starting_agent=mock_agent,
                input=input_text,
                run_config=run_config,
            )
        assert "Unknown next step type" in str(exc_info.value)


@pytest.mark.asyncio
async def test_initialize_run_with_invalid_input(mock_agent):
    """Test run initialization with invalid input."""
    with pytest.raises(RunnerError) as exc_info:
        await Runner._initialize_run(
            starting_agent=mock_agent,
            input=None,  # Invalid input
            context=None,
            max_turns=10,
            charms=None,
            run_config=None,
        )
    assert "Invalid input" in str(exc_info.value)


@pytest.mark.asyncio
async def test_initialize_run_with_invalid_max_turns(mock_agent):
    """Test run initialization with invalid max_turns."""
    with pytest.raises(RunnerError) as exc_info:
        await Runner._initialize_run(
            starting_agent=mock_agent,
            input="test input",
            context=None,
            max_turns=0,  # Invalid max_turns
            charms=None,
            run_config=None,
        )
    assert "Max turns must be positive" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_turn_with_empty_generated_items(mock_agent, context_wrapper, mock_charms):
    """Test running a turn with empty generated items."""
    mock_turn_result = SingleStepResult(
        next_step=NextStepFinalOutput(output="test output"),
        model_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        original_input="test input",
        pre_step_items=[],
        new_step_items=[],
    )

    with patch("src.runners.run.Runner._run_turn", return_value=mock_turn_result):
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
        assert len(result.new_step_items) == 0


@pytest.mark.asyncio
async def test_run_turn_with_multiple_generated_items(
    mock_agent, context_wrapper, mock_charms, message_output
):
    """Test running a turn with multiple generated items."""
    mock_turn_result = SingleStepResult(
        next_step=NextStepFinalOutput(output="test output"),
        model_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        original_input="test input",
        pre_step_items=[],
        new_step_items=[
            MessageOutputItem(agent=mock_agent, raw_item=message_output),
            MessageOutputItem(agent=mock_agent, raw_item=message_output),
        ],
    )

    with patch("src.runners.run.Runner._run_turn", return_value=mock_turn_result):
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
        assert len(result.new_step_items) == 2


@pytest.mark.asyncio
async def test_run_streamed_with_partial_responses(mock_agent, mock_model_client):
    """Test streamed agent run with partial responses."""
    # Create a mock streaming result with partial responses
    mock_streaming_result = RunResultStreaming(
        input="test input",
        new_items=[],
        current_agent=mock_agent,
        raw_responses=[
            ModelResponse(
                output=[{"type": "partial", "content": "partial response"}],
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
                referenceable_id="test_id",
            )
        ],
        final_output="test output",
        is_complete=True,
        current_turn=1,
        max_turns=10,
        input_shield_results=[],
        output_shield_results=[],
    )

    with patch("src.runners.run.Runner.run_streamed", return_value=mock_streaming_result):
        result = await Runner.run_streamed(starting_agent=mock_agent, input="test input")

        assert isinstance(result, RunResultStreaming)
        assert result.final_output == "test output"
        assert len(result.raw_responses) == 1
        assert result.raw_responses[0].output[0]["type"] == "partial"


@pytest.mark.asyncio
async def test_run_streamed_with_multiple_turns(mock_agent, mock_model_client):
    """Test streamed agent run with multiple turns."""
    # Create a mock streaming result with multiple turns
    mock_streaming_result = RunResultStreaming(
        input="test input",
        new_items=[],
        current_agent=mock_agent,
        raw_responses=[
            ModelResponse(
                output=[{"type": "message", "content": "turn 1 response"}],
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
                referenceable_id="test_id_1",
            ),
            ModelResponse(
                output=[{"type": "message", "content": "turn 2 response"}],
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
                referenceable_id="test_id_2",
            ),
        ],
        final_output="final output",
        is_complete=True,
        current_turn=2,
        max_turns=10,
        input_shield_results=[],
        output_shield_results=[],
    )

    with patch("src.runners.run.Runner.run_streamed", return_value=mock_streaming_result):
        result = await Runner.run_streamed(starting_agent=mock_agent, input="test input")

        assert isinstance(result, RunResultStreaming)
        assert result.final_output == "final output"
        assert len(result.raw_responses) == 2
        assert result.current_turn == 2


def test_run_sync_with_error(mock_agent):
    """Test synchronous agent run with error."""
    # Create a new event loop for the test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Patch the Runner.run method to raise an error
        with patch("src.runners.run.Runner.run", side_effect=RunnerError("Test error")):
            with pytest.raises(RunnerError) as exc_info:
                Runner.run_sync(starting_agent=mock_agent, input="test input")
            assert "Test error" in str(exc_info.value)
    finally:
        # Clean up the event loop
        loop.close()
        asyncio.set_event_loop(None)


def test_run_sync_with_timeout(mock_agent):
    """Test synchronous agent run with timeout."""
    # Create a new event loop for the test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Patch the Runner.run method to simulate a timeout
        async def mock_run(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate a long-running operation
            return RunResult(
                input="test input",
                final_output="test output",
                raw_responses=[],
                new_items=[],
                input_shield_results=[],
                output_shield_results=[],
                _last_agent=mock_agent,
            )

        with patch("src.runners.run.Runner.run", side_effect=mock_run):
            # Set a short timeout
            with pytest.raises(asyncio.TimeoutError):
                Runner.run_sync(starting_agent=mock_agent, input="test input", timeout=0.1)
    finally:
        # Clean up the event loop
        loop.close()
        asyncio.set_event_loop(None)


@pytest.mark.asyncio
async def test_run_shields_with_empty_shields(mock_agent, context_wrapper):
    """Test running shields with empty shield list."""
    result = await Runner._run_shields(
        agent=mock_agent,
        input_or_output="test input",
        context=context_wrapper,
        shields=[],
        is_input=True,
    )
    assert len(result) == 0


@pytest.mark.asyncio
async def test_run_shields_with_multiple_shields(mock_agent, context_wrapper):
    """Test running shields with multiple shields."""
    # Create mock shields
    mock_shield1 = AsyncMock(spec=InputShield)
    mock_shield1.name = "shield1"
    shield_result1 = InputShieldResult(
        tripwire_triggered=False,
        shield=mock_shield1,
        agent=mock_agent,
        input="test input",
        output=ShieldResult(success=True, data="test input"),
    )
    mock_shield1.run.return_value = shield_result1

    mock_shield2 = AsyncMock(spec=InputShield)
    mock_shield2.name = "shield2"
    shield_result2 = InputShieldResult(
        tripwire_triggered=False,
        shield=mock_shield2,
        agent=mock_agent,
        input="test input",
        output=ShieldResult(success=True, data="test input"),
    )
    mock_shield2.run.return_value = shield_result2

    result = await Runner._run_shields(
        agent=mock_agent,
        input_or_output="test input",
        context=context_wrapper,
        shields=[mock_shield1, mock_shield2],
        is_input=True,
    )

    assert len(result) == 2
    assert result[0].shield.name == "shield1"
    assert result[1].shield.name == "shield2"
    assert mock_shield1.run.called
    assert mock_shield2.run.called


@pytest.mark.asyncio
async def test_run_shields_with_shield_error(mock_agent, context_wrapper):
    """Test running shields with shield error."""
    # Create a mock shield that will raise an error
    mock_shield = AsyncMock(spec=InputShield)
    mock_shield.name = "error_shield"
    mock_shield.run.side_effect = Exception("Shield error")

    with pytest.raises(RunnerError) as exc_info:
        await Runner._run_shields(
            agent=mock_agent,
            input_or_output="test input",
            context=context_wrapper,
            shields=[mock_shield],
            is_input=True,
        )
    assert "Shield error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_input_shields_with_queue(mock_agent, context_wrapper):
    """Test running input shields with queue."""
    # Create a mock shield
    mock_shield = AsyncMock(spec=InputShield)
    mock_shield.name = "test_shield"
    shield_result = InputShieldResult(
        tripwire_triggered=False,
        shield=mock_shield,
        agent=mock_agent,
        input="test input",
        output=ShieldResult(success=True, data="test input"),
    )
    mock_shield.run.return_value = shield_result

    # Create a mock streaming result
    streamed_result = RunResultStreaming(
        input="test input",
        new_items=[],
        current_agent=mock_agent,
        raw_responses=[],
        final_output=None,
        is_complete=False,
        current_turn=0,
        max_turns=10,
        input_shield_results=[],
        output_shield_results=[],
    )

    # Run the test
    await Runner._run_input_shields_with_queue(
        agent=mock_agent,
        input="test input",
        context=context_wrapper,
        shields=[mock_shield],
        streamed_result=streamed_result,
    )

    assert len(streamed_result.input_shield_results) == 1
    assert streamed_result.input_shield_results[0].output.success


@pytest.mark.asyncio
async def test_run_shields(mock_agent, context_wrapper):
    """Test running shields."""
    # Create a mock shield
    mock_shield = AsyncMock(spec=InputShield)
    mock_shield.name = "test_shield"
    shield_result = InputShieldResult(
        tripwire_triggered=False,
        shield=mock_shield,
        agent=mock_agent,
        input="test input",
        output=ShieldResult(success=True, data="test input"),
    )
    mock_shield.run.return_value = shield_result

    # Run the test
    results = await Runner._run_shields(
        agent=mock_agent,
        input_or_output="test input",
        context=context_wrapper,
        shields=[mock_shield],
        is_input=True,
    )

    assert len(results) == 1
    assert results[0].output.success


@pytest.mark.asyncio
async def test_handle_streamed_error():
    """Test handling streamed error."""
    # Create a mock streaming result
    streamed_result = RunResultStreaming(
        input="test input",
        new_items=[],
        current_agent=MagicMock(),
        raw_responses=[],
        final_output=None,
        is_complete=False,
        current_turn=0,
        max_turns=10,
        input_shield_results=[],
        output_shield_results=[],
    )

    # Run the test
    Runner._handle_streamed_error(streamed_result)

    assert streamed_result.is_complete


@pytest.mark.asyncio
async def test_handle_streamed_final_output(mock_agent, context_wrapper):
    """Test handling streamed final output."""
    # Create a mock streaming result
    streamed_result = RunResultStreaming(
        input="test input",
        new_items=[],
        current_agent=mock_agent,
        raw_responses=[],
        final_output=None,
        is_complete=False,
        current_turn=0,
        max_turns=10,
        input_shield_results=[],
        output_shield_results=[],
    )

    # Create a mock turn result
    turn_result = SingleStepResult(
        next_step=NextStepFinalOutput(output="test output"),
        model_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        original_input="test input",
        pre_step_items=[],
        new_step_items=[],
    )

    # Run the test
    await Runner._handle_streamed_final_output(
        current_agent=mock_agent,
        turn_result=turn_result,
        streamed_result=streamed_result,
        context_wrapper=context_wrapper,
        run_config=RunConfig(),
    )

    assert streamed_result.final_output == "test output"


def test_update_streamed_result():
    """Test updating streamed result."""
    # Create a mock streaming result
    streamed_result = RunResultStreaming(
        input="test input",
        new_items=[],
        current_agent=MagicMock(),
        raw_responses=[],
        final_output=None,
        is_complete=False,
        current_turn=0,
        max_turns=10,
        input_shield_results=[],
        output_shield_results=[],
    )

    # Create a mock turn result
    turn_result = SingleStepResult(
        next_step=NextStepFinalOutput(output="test output"),
        model_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        original_input="new input",
        pre_step_items=[],
        new_step_items=[],
    )

    # Run the test
    Runner._update_streamed_result(streamed_result, turn_result)

    assert streamed_result.input == "new input"
    assert len(streamed_result.raw_responses) == 1


@pytest.mark.asyncio
async def test_queue_event():
    """Test queueing an event."""
    # Create a mock queue
    queue = asyncio.Queue()

    # Create a test event
    event = {"type": "test", "data": "test data"}

    # Run the test
    await Runner._queue_event(queue, event)

    # Verify the event was queued
    queued_event = await queue.get()
    assert queued_event == event
