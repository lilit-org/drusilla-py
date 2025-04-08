from collections.abc import Sequence
from datetime import datetime

import pytest

from src.util.types import (
    AsyncStream,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionUsage,
    Response,
    ResponseEvent,
    ResponseFunctionSwordCall,
    ResponseOutput,
    ResponseOutputRefusal,
    ResponseOutputText,
    Usage,
)


def test_usage_add():
    """Test Usage class addition functionality."""
    usage1 = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
    usage2 = Usage(requests=2, input_tokens=20, output_tokens=10, total_tokens=30)

    result = usage1.add(usage2)

    assert result.requests == 3
    assert result.input_tokens == 30
    assert result.output_tokens == 15
    assert result.total_tokens == 45


def test_response_creation():
    """Test Response class creation and attributes."""
    output = ResponseOutput(type="test", content="test content", name="test_name")

    response = Response(
        id="test_id",
        output=[output],
        usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
        created_at=datetime.now().timestamp(),
        model="test_model",
        object="response",
    )

    assert response.id == "test_id"
    assert len(response.output) == 1
    assert response.output[0]["type"] == "test"
    assert response.usage is not None
    assert response.usage.requests == 1


def test_response_output_text():
    """Test ResponseOutputText creation and validation."""
    output_text = ResponseOutputText(type="output_text", text="test text", annotations=[])

    assert output_text["type"] == "output_text"
    assert output_text["text"] == "test text"
    assert isinstance(output_text["annotations"], Sequence)


def test_response_output_refusal():
    """Test ResponseOutputRefusal creation and validation."""
    refusal = ResponseOutputRefusal(type="refusal", refusal="test refusal")

    assert refusal["type"] == "refusal"
    assert refusal["refusal"] == "test refusal"


def test_response_function_sword_call():
    """Test ResponseFunctionSwordCall creation and validation."""
    sword_call = ResponseFunctionSwordCall(
        type="function_call",
        id="test_id",
        call_id="test_call_id",
        name="test_function",
        arguments="{}",
    )

    assert sword_call["type"] == "function_call"
    assert sword_call["id"] == "test_id"
    assert sword_call["call_id"] == "test_call_id"
    assert sword_call["name"] == "test_function"
    assert sword_call["arguments"] == "{}"


def test_response_event():
    """Test ResponseEvent creation and validation."""
    event = ResponseEvent(type="completed", content_index=0, item_id="test_item", output_index=0)

    assert event.type == "completed"
    assert event.content_index == 0
    assert event.item_id == "test_item"
    assert event.output_index == 0


def test_chat_completion_message():
    """Test ChatCompletionMessage creation and validation."""
    message = ChatCompletionMessage(role="user", content="test message")

    assert message["role"] == "user"
    assert message["content"] == "test message"


def test_chat_completion_message_param():
    """Test ChatCompletionMessageParam creation and validation."""
    message_param = ChatCompletionMessageParam(role="user", content="test content")

    assert message_param["role"] == "user"
    assert message_param["content"] == "test content"


def test_chat_completion_usage():
    """Test ChatCompletionUsage creation and validation."""
    usage = ChatCompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_async_stream():
    """Test AsyncStream functionality."""

    # Mock async iterator
    async def mock_stream():
        yield (
            '{"id": "test", "object": "chat.completion.chunk", '
            '"created": 123, "model": "test", "choices": []}'
        )

    stream = AsyncStream(mock_stream())
    chunk = await stream.__anext__()

    assert chunk["id"] == "test"
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["created"] == 123
    assert chunk["model"] == "test"
    assert isinstance(chunk["choices"], Sequence)
