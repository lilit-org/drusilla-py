from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.output import AgentOutputSchema
from src.gear.orbs import Orbs
from src.gear.sword import Sword
from src.models.chat import ModelChatCompletionsModel, SwordConverter, _Converter, _StreamingState
from src.models.settings import ModelSettings
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
    mock_orb.name = "test_orb"
    mock_orb.sword_name = "test_orb_sword"
    mock_orb.sword_description = "Test orb description"
    mock_orb.input_json_schema = {"type": "object", "properties": {}}
    result = SwordConverter.convert_orb_sword(mock_orb)

    assert isinstance(result, dict)
    assert result["name"] == "test_orb_sword"
    assert result["description"] == "Test orb description"
    assert result["parameters"] == {"type": "object", "properties": {}}
