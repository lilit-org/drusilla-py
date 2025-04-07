import asyncio
from unittest.mock import MagicMock

import pytest

from src.gear.shield import InputShieldResult, OutputShieldResult, ShieldResult
from src.util._items import MessageOutputItem, ModelResponse
from src.util._result import RunResult, RunResultStreaming
from src.util._types import QueueCompleteSentinel, Usage


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.name = "test_agent"
    return agent


@pytest.fixture
def mock_input_shield_result():
    shield = MagicMock()
    shield.name = "test_shield"
    agent = MagicMock()
    agent.name = "test_agent"
    return InputShieldResult(
        tripwire_triggered=False,
        shield=shield,
        agent=agent,
        input="test input",
        output=ShieldResult(tripwire_triggered=False, result="processed"),
    )


@pytest.fixture
def mock_output_shield_result():
    shield = MagicMock()
    shield.name = "test_shield"
    agent = MagicMock()
    agent.name = "test_agent"
    return OutputShieldResult(
        tripwire_triggered=False,
        shield=shield,
        agent=agent,
        agent_output="test output",
        output=ShieldResult(tripwire_triggered=False, result="processed"),
    )


@pytest.fixture
def mock_model_response():
    return ModelResponse(
        output=[{"type": "message", "content": "Test response"}],
        usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
        referenceable_id="test_id",
    )


@pytest.fixture
def mock_run_item(mock_agent):
    return MessageOutputItem(agent=mock_agent, raw_item={"type": "message", "content": "Test item"})


@pytest.fixture
async def mock_streaming_result(mock_agent):
    """Fixture that creates a RunResultStreaming instance and ensures cleanup."""
    result = RunResultStreaming(
        input="test input",
        new_items=[],
        raw_responses=[],
        final_output=None,
        input_shield_results=[],
        output_shield_results=[],
        current_agent=mock_agent,
    )
    yield result
    # Ensure cleanup
    result.is_complete = True
    result._cleanup_tasks()
    # Clear the event queue
    while not result._event_queue.empty():
        try:
            result._event_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    # Add sentinel to ensure stream terminates
    try:
        result._event_queue.put_nowait(QueueCompleteSentinel())
    except asyncio.QueueFull:
        pass


def test_run_result_initialization(
    mock_agent,
    mock_input_shield_result,
    mock_output_shield_result,
    mock_model_response,
    mock_run_item,
):
    """Test RunResult initialization and properties."""
    result = RunResult(
        input="test input",
        new_items=[mock_run_item],
        raw_responses=[mock_model_response],
        final_output="test output",
        input_shield_results=[mock_input_shield_result],
        output_shield_results=[mock_output_shield_result],
        _last_agent=mock_agent,
    )

    assert result.input == "test input"
    assert len(result.new_items) == 1
    assert len(result.raw_responses) == 1
    assert result.final_output == "test output"
    assert len(result.input_shield_results) == 1
    assert len(result.output_shield_results) == 1
    assert result.last_agent == mock_agent
    assert "test_agent" in str(result)


@pytest.mark.asyncio
async def test_run_result_streaming_initialization(mock_streaming_result):
    """Test RunResultStreaming initialization and basic properties."""
    result = mock_streaming_result

    assert result.input == "test input"
    assert len(result.new_items) == 0
    assert len(result.raw_responses) == 0
    assert result.final_output is None
    assert len(result.input_shield_results) == 0
    assert len(result.output_shield_results) == 0
    assert result.current_agent.name == "test_agent"
    assert not result.is_complete
    assert result.current_turn == 0


@pytest.mark.asyncio
async def test_run_result_streaming_events(mock_streaming_result):
    """Test RunResultStreaming event streaming functionality."""
    result = mock_streaming_result

    # Add some test events to the queue
    test_event = MagicMock()
    result._event_queue.put_nowait(test_event)
    result._event_queue.put_nowait(QueueCompleteSentinel())

    # Test streaming events
    events = []
    async for event in result.stream_events():
        events.append(event)

    assert len(events) == 1
    assert events[0] == test_event
    assert result.is_complete


@pytest.mark.asyncio
async def test_run_result_streaming_cleanup(mock_streaming_result):
    """Test RunResultStreaming task cleanup."""
    result = mock_streaming_result

    # Create mock tasks
    result._run_impl_task = MagicMock()
    result._input_shields_task = MagicMock()
    result._output_shields_task = MagicMock()

    # Set up the mock tasks to be not done
    result._run_impl_task.done.return_value = False
    result._input_shields_task.done.return_value = False
    result._output_shields_task.done.return_value = False

    # Test cleanup
    result._cleanup_tasks()

    # Verify tasks were cancelled
    result._run_impl_task.cancel.assert_called_once()
    result._input_shields_task.cancel.assert_called_once()
    result._output_shields_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_run_result_streaming_exception_handling(mock_streaming_result):
    """Test RunResultStreaming exception handling during streaming."""
    result = mock_streaming_result

    # Set an exception
    test_exception = Exception("Test error")
    result._stored_exception = test_exception

    # Test that the exception is raised during streaming
    with pytest.raises(Exception) as exc_info:
        async for _ in result.stream_events():
            pass

    assert exc_info.value == test_exception
    assert result.is_complete


def test_run_result_str_representation(
    mock_agent,
    mock_input_shield_result,
    mock_output_shield_result,
    mock_model_response,
    mock_run_item,
):
    """Test string representation of RunResult."""
    result = RunResult(
        input="test input",
        new_items=[mock_run_item],
        raw_responses=[mock_model_response],
        final_output="test output",
        input_shield_results=[mock_input_shield_result],
        output_shield_results=[mock_output_shield_result],
        _last_agent=mock_agent,
    )

    str_rep = str(result)
    assert "RunResult" in str_rep
    assert "test_agent" in str_rep
    assert "1 items" in str_rep
    assert "1 responses" in str_rep
    assert "Complete" in str_rep
    assert "test output" in str_rep


@pytest.mark.asyncio
async def test_run_result_streaming_str_representation(mock_streaming_result):
    """Test string representation of RunResultStreaming."""
    result = mock_streaming_result
    result.final_output = "test output"

    str_rep = str(result)
    assert "RunResultStreaming" in str_rep
    assert "test_agent" in str_rep
    assert "0 items" in str_rep
    assert "0 responses" in str_rep
    assert "In Progress" in str_rep
    assert "test output" in str_rep


@pytest.mark.asyncio
async def test_run_result_streaming_queue_limits(mock_streaming_result):
    """Test RunResultStreaming queue size limits."""
    result = mock_streaming_result

    # Override the queue size for testing
    result._event_queue = asyncio.Queue(maxsize=5)

    # Fill up the queue
    for _i in range(5):
        result._event_queue.put_nowait(MagicMock())

    # Verify queue is full
    assert result._event_queue.full()

    # Try to add one more item - should raise QueueFull
    with pytest.raises(asyncio.QueueFull):
        result._event_queue.put_nowait(MagicMock())


@pytest.mark.asyncio
async def test_run_result_streaming_last_agent(mock_streaming_result):
    """Test last_agent property of RunResultStreaming."""
    result = mock_streaming_result

    assert result.last_agent.name == "test_agent"
