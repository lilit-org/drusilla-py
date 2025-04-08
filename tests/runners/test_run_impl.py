"""Unit tests for the RunImpl class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.agents.agent_v1 import AgentV1 as Agent
from src.gear.charms import RunCharms
from src.gear.orbs import Orbs
from src.gear.sword import Sword
from src.models.chat import ModelChatCompletionsModel
from src.models.interface import Model
from src.models.settings import ModelSettings
from src.runners.items import (
    MessageOutputItem,
    ModelResponse,
    OrbsOutputItem,
    SwordCallOutputItem,
    Usage,
)
from src.runners.run import RunContextWrapper
from src.runners.run_impl import (
    NextStepFinalOutput,
    NextStepOrbs,
    NextStepRunAgain,
    ProcessedResponse,
    ResponseFunctionSwordCall,
    RunImpl,
    SingleStepResult,
    SwordResult,
    SwordRunFunction,
    SwordRunOrbs,
    SwordsToFinalOutputResult,
)
from src.util.exceptions import AgentError, GenericError


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
def mock_sword():
    sword = AsyncMock(spec=Sword)
    sword.name = "test_sword"
    sword.on_invoke_sword = AsyncMock(return_value="sword_output")
    sword.sword_use_behavior = AsyncMock(
        return_value=SwordsToFinalOutputResult(is_final_output=False, final_output=None)
    )
    return sword


@pytest.fixture
def mock_orb():
    orb = AsyncMock(spec=Orbs)
    orb.name = "test_orb"
    orb.execute = AsyncMock(return_value="orb_output")
    orb.on_invoke_orbs = AsyncMock(return_value="orb_output")
    orb.get_transfer_message = AsyncMock(return_value="transfer message")
    return orb


@pytest.fixture
def mock_charms():
    charms = RunCharms()
    charms.on_sword_start = AsyncMock(return_value="start")
    charms.on_sword_end = AsyncMock(return_value="end")
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
    mock_response.aiter_lines = AsyncMock()
    mock_response.raise_for_status = AsyncMock()

    mock_http_client = AsyncMock()
    mock_http_client.post = AsyncMock(return_value=mock_response)

    mock_deepseek = AsyncMock()
    mock_deepseek.http_client = mock_http_client
    mock_deepseek.base_url = "http://localhost:11434"
    mock_deepseek.api_key = None

    mock_provider = AsyncMock()
    mock_provider._get_client.return_value = mock_deepseek
    mock_provider.get_model.return_value = ModelChatCompletionsModel(
        model="test-model", model_client=mock_deepseek
    )

    monkeypatch.setattr("src.models.provider.ModelProvider", mock_provider)
    return mock_deepseek


@pytest.fixture
def sword_call():
    return {
        "type": "function_call",
        "id": "test_id",
        "call_id": "test_id",
        "name": "test_sword",
        "arguments": "{}",
    }


@pytest.fixture
def orb_call():
    return {
        "type": "function_call",
        "id": "test_id",
        "call_id": "test_id",
        "name": "test_orb",
        "arguments": "{}",
    }


# Test cases
@pytest.mark.asyncio
async def test_process_model_response(
    mock_agent: Agent, message_output: dict, mock_model_client: AsyncMock
) -> None:
    """Test processing model response."""
    model_response = ModelResponse(output=[message_output], usage=None, referenceable_id=None)

    processed_response = RunImpl.process_model_response(
        agent=mock_agent, response=model_response, output_schema=None, orbs=[]
    )

    assert processed_response is not None
    assert isinstance(processed_response, ProcessedResponse)
    assert len(processed_response.new_items) == 1
    assert isinstance(processed_response.new_items[0], MessageOutputItem)
    assert processed_response.orbs == []
    assert processed_response.functions == []


@pytest.mark.asyncio
async def test_execute_sword_call(
    mock_agent: Agent,
    mock_sword: AsyncMock,
    mock_charms: RunCharms,
    context_wrapper: RunContextWrapper,
    sword_call: ResponseFunctionSwordCall,
) -> None:
    """Test executing sword call."""
    mock_sword.on_invoke_sword = AsyncMock(return_value="sword_output")
    mock_sword.sword_use_behavior = AsyncMock(
        return_value=SwordsToFinalOutputResult(is_final_output=False, final_output=None)
    )
    mock_agent.sword_use_behavior = "run_llm_again"  # Set the agent's sword_use_behavior

    next_step = await RunImpl.execute_swords_and_side_effects(
        agent=mock_agent,
        original_input="test input",
        pre_step_items=[],
        new_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        processed_response=ProcessedResponse(
            new_items=[],
            orbs=[],
            functions=[SwordRunFunction(sword_call=sword_call, function_sword=mock_sword)],
        ),
        output_schema=None,
        charms=mock_charms,
        context_wrapper=context_wrapper,
        run_config=None,
    )

    assert isinstance(next_step, SingleStepResult)
    assert isinstance(next_step.next_step, NextStepRunAgain)
    mock_sword.on_invoke_sword.assert_called_once()


@pytest.mark.asyncio
async def test_execute_orb_call(
    mock_agent: Agent,
    mock_orb: AsyncMock,
    mock_charms: RunCharms,
    context_wrapper: RunContextWrapper,
    orb_call: ResponseFunctionSwordCall,
) -> None:
    """Test executing orb call."""
    mock_orb.on_invoke_orbs = AsyncMock(return_value=mock_agent)
    mock_orb.input_filter = None  # Explicitly set input_filter to None
    mock_orb.get_transfer_message = Mock(return_value="transfer message")

    next_step = await RunImpl.execute_orbs(
        agent=mock_agent,
        original_input="test input",
        pre_step_items=[],
        new_step_items=[],
        new_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        run_orbs=[SwordRunOrbs(orbs=mock_orb, sword_call=orb_call)],
        charms=mock_charms,
        context_wrapper=context_wrapper,
        run_config=None,
    )

    assert isinstance(next_step, SingleStepResult)
    assert isinstance(next_step.next_step, NextStepOrbs)
    assert next_step.next_step.new_agent == mock_agent
    mock_orb.on_invoke_orbs.assert_called_once()


@pytest.mark.asyncio
async def test_run_single_turn(
    mock_agent: Agent, mock_charms: RunCharms, context_wrapper: RunContextWrapper
) -> None:
    """Test running a single turn."""
    result = await RunImpl.execute_swords_and_side_effects(
        agent=mock_agent,
        original_input="test input",
        pre_step_items=[],
        new_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        processed_response=ProcessedResponse(new_items=[], orbs=[], functions=[]),
        output_schema=None,
        charms=mock_charms,
        context_wrapper=context_wrapper,
        run_config=None,
    )

    assert isinstance(result, SingleStepResult)
    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == ""  # Empty output is expected when no functions are executed


@pytest.mark.asyncio
async def test_run_turn_error(
    mock_agent: Agent, mock_charms: RunCharms, context_wrapper: RunContextWrapper
) -> None:
    """Test error handling in run turn."""
    with patch("src.runners.run_impl.RunImpl.execute_swords_and_side_effects") as mock_execute:
        mock_execute.side_effect = GenericError("Test error")

        with pytest.raises(GenericError):
            await RunImpl.execute_swords_and_side_effects(
                agent=mock_agent,
                original_input="test input",
                pre_step_items=[],
                new_response=ModelResponse(output=[], usage=None, referenceable_id=None),
                processed_response=ProcessedResponse(new_items=[], orbs=[], functions=[]),
                output_schema=None,
                charms=mock_charms,
                context_wrapper=context_wrapper,
                run_config=None,
            )


@pytest.mark.asyncio
async def test_execute_function_sword(
    mock_agent: Agent,
    mock_sword: AsyncMock,
    mock_charms: RunCharms,
    context_wrapper: RunContextWrapper,
    sword_call: ResponseFunctionSwordCall,
) -> None:
    """Test execution of function sword calls."""
    sword_run = SwordRunFunction(sword_call=sword_call, function_sword=mock_sword)

    mock_sword.on_invoke_sword = AsyncMock(return_value="test_result")
    mock_charms.on_sword_start = AsyncMock(return_value="start")
    mock_charms.on_sword_end = AsyncMock(return_value="end")
    mock_agent.charms = None

    results = await RunImpl.execute_function_sword_calls(
        agent=mock_agent,
        sword_runs=[sword_run],
        charms=mock_charms,
        context_wrapper=context_wrapper,
    )

    assert len(results) == 1
    assert isinstance(results[0], SwordResult)
    assert results[0].output == "test_result"
    assert results[0].sword == mock_sword
    assert isinstance(results[0].run_item, SwordCallOutputItem)
    assert results[0].run_item.output == "test_result"
    assert results[0].run_item.raw_item["call_id"] == "test_id"


@pytest.mark.asyncio
async def test_execute_function_sword_error(
    mock_agent: Agent,
    mock_sword: AsyncMock,
    mock_charms: RunCharms,
    context_wrapper: RunContextWrapper,
    sword_call: ResponseFunctionSwordCall,
) -> None:
    """Test error handling in function sword execution."""
    sword_run = SwordRunFunction(sword_call=sword_call, function_sword=mock_sword)

    mock_sword.on_invoke_sword = AsyncMock(side_effect=GenericError("Test error"))
    mock_charms.on_sword_start = AsyncMock(return_value="start")
    mock_agent.charms = None

    with pytest.raises(AgentError):
        await RunImpl.execute_function_sword_calls(
            agent=mock_agent,
            sword_runs=[sword_run],
            charms=mock_charms,
            context_wrapper=context_wrapper,
        )


@pytest.mark.asyncio
async def test_execute_orbs(
    mock_agent: Agent,
    mock_orb: AsyncMock,
    mock_charms: RunCharms,
    context_wrapper: RunContextWrapper,
    orb_call: ResponseFunctionSwordCall,
) -> None:
    """Test execution of orbs."""
    orbs_run = SwordRunOrbs(orbs=mock_orb, sword_call=orb_call)

    mock_orb.on_invoke_orbs = AsyncMock(return_value=mock_agent)
    mock_orb.get_transfer_message = Mock(return_value="transfer message")
    mock_orb.input_filter = None
    mock_charms.on_orbs = AsyncMock(return_value="orbs_result")

    result = await RunImpl.execute_orbs(
        agent=mock_agent,
        original_input="test input",
        pre_step_items=[],
        new_step_items=[],
        new_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        run_orbs=[orbs_run],
        charms=mock_charms,
        context_wrapper=context_wrapper,
        run_config=None,
    )

    assert isinstance(result, SingleStepResult)
    assert len(result.new_step_items) == 1
    assert isinstance(result.new_step_items[0], OrbsOutputItem)
    mock_orb.on_invoke_orbs.assert_called_once_with(context_wrapper, orb_call["arguments"])
    mock_charms.on_orbs.assert_called_once_with(
        context=context_wrapper, from_agent=mock_agent, to_agent=mock_agent
    )


@pytest.mark.asyncio
async def test_execute_final_output(
    mock_agent: Agent, context_wrapper: RunContextWrapper, mock_charms: RunCharms
) -> None:
    """Test execution of final output."""
    result = await RunImpl.execute_final_output(
        agent=mock_agent,
        original_input="test input",
        new_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        pre_step_items=[],
        new_step_items=[],
        final_output="test output",
        charms=mock_charms,
        context_wrapper=context_wrapper,
    )

    assert isinstance(result, SingleStepResult)
    assert isinstance(result.next_step, NextStepFinalOutput)
    assert result.next_step.output == "test output"


@pytest.mark.asyncio
async def test_check_final_output_from_swords(
    mock_agent: Agent, context_wrapper: RunContextWrapper
) -> None:
    """Test checking for final output from sword results."""
    sword_result = MagicMock()
    sword_result.sword = MagicMock()
    sword_result.sword.name = "test_sword"
    sword_result.output = "test output"
    mock_agent.sword_use_behavior = "run_llm_again"  # Set the agent's sword_use_behavior

    result = await RunImpl._check_for_final_output_from_swords(
        agent=mock_agent, sword_results=[sword_result], context_wrapper=context_wrapper
    )

    assert result.is_final_output is False
    assert result.final_output is None


@pytest.mark.asyncio
async def test_stream_step_result(mock_agent: Agent) -> None:
    """Test streaming step results to queue."""
    message_output = {
        "type": "message",
        "content": [{"type": "output_text", "text": "test output"}],
    }

    step_result = SingleStepResult(
        original_input="test input",
        model_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        pre_step_items=[],
        new_step_items=[MessageOutputItem(agent=mock_agent, raw_item=message_output)],
        next_step=NextStepRunAgain(),
    )

    queue = AsyncMock()
    await RunImpl.stream_step_result_to_queue(step_result, queue)

    assert queue.put.called


@pytest.mark.asyncio
async def test_stream_step_result_error(mock_agent: Agent) -> None:
    """Test error handling in streaming step results."""
    message_output = {
        "type": "message",
        "content": [{"type": "output_text", "text": "test output"}],
    }

    step_result = SingleStepResult(
        original_input="test input",
        model_response=ModelResponse(output=[], usage=None, referenceable_id=None),
        pre_step_items=[],
        new_step_items=[MessageOutputItem(agent=mock_agent, raw_item=message_output)],
        next_step=NextStepRunAgain(),
    )

    queue = AsyncMock()
    queue.put.side_effect = GenericError("Test error")

    with pytest.raises(GenericError):
        await RunImpl.stream_step_result_to_queue(step_result, queue)


@pytest.mark.asyncio
async def test_execute_orbs_error(
    mock_agent: Agent,
    mock_orb: AsyncMock,
    mock_charms: RunCharms,
    context_wrapper: RunContextWrapper,
    orb_call: ResponseFunctionSwordCall,
) -> None:
    """Test error handling in orb execution."""
    orbs_run = SwordRunOrbs(orbs=mock_orb, sword_call=orb_call)

    mock_orb.on_invoke_orbs = AsyncMock(side_effect=GenericError("Test error"))

    with pytest.raises(GenericError):
        await RunImpl.execute_orbs(
            agent=mock_agent,
            original_input="test input",
            pre_step_items=[],
            new_step_items=[],
            new_response=ModelResponse(output=[], usage=None, referenceable_id=None),
            run_orbs=[orbs_run],
            charms=mock_charms,
            context_wrapper=context_wrapper,
            run_config=None,
        )


@pytest.mark.asyncio
async def test_run_impl_with_error_handling(context_wrapper, mock_agent):
    """Test run implementation with error handling."""
    from src.runners.run_impl import RunImpl
    from src.util.exceptions import ModelError

    # Create a mock model that raises an error
    class ErrorModel:
        async def get_response(self, *args, **kwargs):
            raise ModelError("Test error")

    agent = mock_agent.clone(model=ErrorModel())

    with pytest.raises(ModelError):
        await RunImpl.run(agent=agent, input="test input", context=context_wrapper, max_turns=1)


@pytest.mark.asyncio
async def test_run_impl_with_custom_output_type(context_wrapper, mock_agent):
    """Test run implementation with custom output type."""
    from src.runners.items import ModelResponse, Usage
    from src.runners.run_impl import RunImpl

    # Create a mock model that returns a specific output type
    class CustomOutputModel:
        async def get_response(self, *args, **kwargs):
            return ModelResponse(
                output=[
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "custom output"}],
                        "role": "assistant",
                    }
                ],
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
                referenceable_id="test_id",
            )

    # Create an agent with custom output type
    agent = mock_agent.clone()
    agent.model = CustomOutputModel()
    agent.output_type = str
    agent.charms = RunCharms()
    agent.charms.on_end = AsyncMock(return_value=None)

    result = await RunImpl.run(
        agent=agent, input="test input", context=context_wrapper, max_turns=1
    )

    assert result == "custom output"


@pytest.mark.asyncio
async def test_run_impl_with_multiple_turns(context_wrapper, mock_agent):
    """Test run implementation with multiple turns."""
    from src.runners.items import ModelResponse, Usage
    from src.runners.run_impl import RunImpl

    turn_count = 0

    class MultiTurnModel:
        async def get_response(self, *args, **kwargs):
            nonlocal turn_count
            turn_count += 1
            return ModelResponse(
                output=[
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": f"turn {turn_count}"}],
                        "role": "assistant",
                    }
                ],
                usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
                referenceable_id=f"test_id_{turn_count}",
            )

    agent = mock_agent.clone()
    agent.model = MultiTurnModel()
    agent.output_type = str
    agent.charms = RunCharms()
    agent.charms.on_end = AsyncMock(return_value=None)

    result = await RunImpl.run(
        agent=agent, input="test input", context=context_wrapper, max_turns=3
    )

    assert result == "turn 3"
