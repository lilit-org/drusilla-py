from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.output import AgentOutputSchema
from src.gear.orbs import Orbs
from src.gear.sword import Sword
from src.models.chat import ModelChatCompletionsModel, SwordConverter, _Converter, _StreamingState
from src.models.settings import ModelSettings
from src.util._exceptions import AgentError, UsageError
from src.util._items import ModelResponse
from src.util._types import ResponseStreamEvent


@pytest.fixture
def mock_model_client():
    return AsyncMock()


@pytest.fixture
def chat_model(mock_model_client):
    return ModelChatCompletionsModel("test-model", mock_model_client)


@pytest.fixture
def model_settings():
    return ModelSettings(temperature=0.7, top_p=0.9, max_tokens=100)


@pytest.fixture
def mock_sword():
    sword = MagicMock(spec=Sword)
    sword.name = "test_sword"
    sword.description = "Test sword description"
    sword.strict_json_schema = {"type": "object", "properties": {}}
    sword.params_json_schema = {"type": "object", "properties": {}}
    return sword


@pytest.fixture
def mock_orb():
    orb = MagicMock(spec=Orbs)
    orb.name = "test_orb"
    orb.sword_name = "test_orb_sword"
    return orb


@pytest.mark.asyncio
async def test_chat_model_get_response(chat_model, model_settings, mock_sword, mock_orb):
    # Setup
    system_instructions = "Test instructions"
    input_text = "Test input"
    output_schema = MagicMock(spec=AgentOutputSchema)

    # Mock the response
    mock_response = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    chat_model._fetch_response = AsyncMock(return_value=mock_response)

    # Call the method
    result = await chat_model.get_response(
        system_instructions=system_instructions,
        input=input_text,
        model_settings=model_settings,
        swords=[mock_sword],
        output_schema=output_schema,
        orbs=[mock_orb],
    )

    # Assertions
    assert isinstance(result, ModelResponse)
    assert len(result.output) > 0
    assert result.output[0]["text"] == "Test response"


@pytest.mark.asyncio
async def test_chat_model_stream_response(chat_model, model_settings, mock_sword, mock_orb):
    # Setup
    system_instructions = "Test instructions"
    input_text = "Test input"
    output_schema = MagicMock(spec=AgentOutputSchema)

    # Mock the stream response
    async def mock_stream():
        yield {"choices": [{"delta": {"content": "Chunk 1"}}]}
        yield {"choices": [{"delta": {"content": "Chunk 2"}}]}

    chat_model._fetch_response = AsyncMock(return_value=(None, mock_stream()))

    # Call the method and collect results
    results = []
    async for chunk in chat_model.stream_response(
        system_instructions=system_instructions,
        input=input_text,
        model_settings=model_settings,
        swords=[mock_sword],
        output_schema=output_schema,
        orbs=[mock_orb],
    ):
        results.append(chunk)

    assert len(results) > 0
    assert any(chunk.type == "content_part.added" for chunk in results)
    assert any(chunk.type == "output_text.delta" for chunk in results)
    assert any(chunk.type == "content_part.done" for chunk in results)
    assert any(chunk.type == "completed" for chunk in results)


def test_streaming_state():
    state = _StreamingState()
    assert state.text_content_index_and_output is None

    # Test setting state
    output_text = MagicMock(spec=ResponseStreamEvent, content="Test")
    state.text_content_index_and_output = (0, output_text)
    assert state.text_content_index_and_output == (0, output_text)


def test_converter_convert_sword_choice():
    # Test valid choices
    assert _Converter.convert_sword_choice("auto") == "auto"
    assert _Converter.convert_sword_choice("required") == "required"
    assert _Converter.convert_sword_choice("none") == "none"
    # Test None input
    assert _Converter.convert_sword_choice(None) == "none"


def test_converter_convert_response_format():
    # Test with output schema
    output_schema = MagicMock(spec=AgentOutputSchema)
    format = _Converter.convert_response_format(output_schema)
    assert format == {"type": "json_object"}


def test_converter_message_to_output_items():
    # Test with valid message
    message = {"role": "assistant", "content": "Test content"}
    items = _Converter.message_to_output_items(message)
    assert len(items) > 0
    assert items[0]["text"] == "Test content"
    assert items[0]["type"] == "output_text"


def test_sword_converter_to_api_format(mock_sword):
    # Test sword conversion
    mock_sword.strict_json_schema = {"type": "object", "properties": {}}
    mock_sword.params_json_schema = {"type": "object", "properties": {}}
    result = SwordConverter.to_api_format(mock_sword)
    assert isinstance(result, dict)
    assert result["name"] == "test_sword"
    assert result["description"] == "Test sword description"
    assert result["parameters"] == {"type": "object", "properties": {}}


def test_sword_converter_convert_orb_sword(mock_orb):
    # Test orb conversion
    mock_orb.to_api_format.return_value = {
        "name": "test_orb",
        "description": "Test orb description",
        "parameters": {"type": "object", "properties": {}},
    }
    result = SwordConverter.convert_orb_sword(mock_orb)

    assert isinstance(result, dict)
    assert result["name"] == "test_orb"
    assert result["description"] == "Test orb description"
    assert result["parameters"] == {"type": "object", "properties": {}}


def test_non_null_or_not_given(chat_model):
    assert chat_model._non_null_or_not_given(None) is None
    assert chat_model._non_null_or_not_given(False) is None
    assert chat_model._non_null_or_not_given("") is None
    assert chat_model._non_null_or_not_given(0) is None
    assert chat_model._non_null_or_not_given([]) is None
    assert chat_model._non_null_or_not_given("test") == "test"
    assert chat_model._non_null_or_not_given(123) == 123
    assert chat_model._non_null_or_not_given(["item"]) == ["item"]


@pytest.mark.asyncio
async def test_get_response_error_handling(chat_model, model_settings):
    # Test handling of API error
    chat_model._fetch_response = AsyncMock(side_effect=AgentError("API Error"))

    with pytest.raises(AgentError, match="API Error"):
        await chat_model.get_response(
            system_instructions=None,
            input="test",
            model_settings=model_settings,
            swords=[],
            output_schema=None,
            orbs=[],
        )


@pytest.mark.asyncio
async def test_stream_response_error_handling(chat_model, model_settings):
    # Test handling of streaming error
    async def mock_stream():
        yield {"choices": [{"delta": {"content": "Chunk 1"}}]}
        yield {"error": "Stream Error"}  # This will be handled gracefully

    chat_model._fetch_response = AsyncMock(return_value=(None, mock_stream()))

    results = []
    async for chunk in chat_model.stream_response(
        system_instructions=None,
        input="test",
        model_settings=model_settings,
        swords=[],
        output_schema=None,
        orbs=[],
    ):
        results.append(chunk)

    assert len(results) > 0
    assert any(chunk.type == "content_part.added" for chunk in results)


def test_converter_maybe_message():
    # Test valid message
    valid_msg = {"role": "user", "content": "test"}
    assert _Converter.maybe_message(valid_msg) == valid_msg

    # Test invalid message
    assert _Converter.maybe_message({"invalid": "format"}) is None
    assert _Converter.maybe_message(None) is None


def test_converter_maybe_file_search_call():
    # Test valid file search call
    valid_call = {
        "type": "file_search_call",
        "id": "test_id",
        "queries": ["test"],
        "status": "completed",
    }
    assert _Converter.maybe_file_search_call(valid_call) == valid_call

    # Test invalid call
    assert _Converter.maybe_file_search_call({"type": "other"}) is None
    assert _Converter.maybe_file_search_call(None) is None


def test_converter_maybe_function_sword_call():
    # Test valid function sword call
    valid_call = {"type": "function_call", "call_id": "test_id", "name": "test", "arguments": "{}"}
    assert _Converter.maybe_function_sword_call(valid_call) == valid_call

    # Test invalid call
    assert _Converter.maybe_function_sword_call({"type": "other"}) is None
    assert _Converter.maybe_function_sword_call(None) is None


def test_converter_maybe_function_sword_call_output():
    # Test valid function sword call output
    valid_output = {"type": "function_call_output", "call_id": "test_id", "output": "test"}
    assert _Converter.maybe_function_sword_call_output(valid_output) == valid_output

    # Test invalid output
    assert _Converter.maybe_function_sword_call_output({"type": "other"}) is None
    assert _Converter.maybe_function_sword_call_output(None) is None


def test_converter_maybe_item_reference():
    # Test valid item reference
    valid_ref = {"type": "item_reference", "item_id": "test"}
    assert _Converter.maybe_item_reference(valid_ref) == valid_ref

    # Test invalid reference
    assert _Converter.maybe_item_reference({"type": "other"}) is None
    assert _Converter.maybe_item_reference(None) is None


def test_converter_maybe_response_output_message():
    # Test valid response output message
    valid_msg = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "test"}],
    }
    assert _Converter.maybe_response_output_message(valid_msg) == valid_msg

    # Test invalid message
    assert _Converter.maybe_response_output_message({"type": "other"}) is None
    assert _Converter.maybe_response_output_message(None) is None


def test_converter_extract_text_content():
    # Test string content
    assert _Converter.extract_text_content("test") == "test"

    # Test list content
    content_list = [
        {"type": "input_text", "text": "test1"},
        {"type": "input_text", "text": "test2"},
    ]
    result = _Converter.extract_text_content(content_list)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(item["type"] == "text" for item in result)
    assert "test1" in [item["text"] for item in result]
    assert "test2" in [item["text"] for item in result]

    # Test empty content
    assert _Converter.extract_text_content([]) == []


def test_converter_extract_all_content():
    # Test string content
    assert _Converter.extract_all_content("test") == "test"

    # Test list with mixed content
    content_list = [
        {"type": "input_text", "text": "test1"},
        {"type": "input_image", "image_url": "http://test.com", "detail": "auto"},
        {"type": "input_text", "text": "test2"},
    ]
    result = _Converter.extract_all_content(content_list)
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == {"type": "text", "text": "test1"}
    assert result[1] == {
        "type": "image_url",
        "image_url": {"url": "http://test.com", "detail": "auto"},
    }
    assert result[2] == {"type": "text", "text": "test2"}

    # Test empty content
    assert _Converter.extract_all_content([]) == []

    # Test error cases
    with pytest.raises(UsageError):
        _Converter.extract_all_content([{"type": "input_file", "path": "test.txt"}])

    with pytest.raises(AgentError):
        _Converter.extract_all_content([{"type": "input_image"}])


def test_converter_items_to_messages():
    # Test string input
    assert _Converter.items_to_messages("test") == [{"role": "user", "content": "test"}]

    # Test list of items with messages
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "test1"}]},
        {"type": "function_call", "call_id": "test_id", "name": "test_sword", "arguments": "{}"},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "test2"}],
        },
    ]
    messages = _Converter.items_to_messages(items)
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], list)
    assert messages[0]["content"][0]["text"] == "test1"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "test2"

    # Test list with only function calls
    items = [
        {"type": "function_call", "call_id": "test_id1", "name": "test_sword1", "arguments": "{}"},
        {"type": "function_call", "call_id": "test_id2", "name": "test_sword2", "arguments": "{}"},
    ]
    messages = _Converter.items_to_messages(items)
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert len(messages[0]["sword_calls"]) == 2

    # Test empty input
    assert _Converter.items_to_messages([]) == []

    # Test error cases
    with pytest.raises(UsageError):
        _Converter.items_to_messages([{"type": "item_reference", "item_id": "test"}])

    with pytest.raises(UsageError):
        _Converter.items_to_messages([{"type": "unknown"}])
