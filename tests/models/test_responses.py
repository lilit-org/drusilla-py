from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.agent_v1 import AgentV1OutputSchema as AgentOutputSchema
from src.gear.orbs import Orbs
from src.gear.sword import Sword
from src.models.responses import Converter, ModelResponsesModel
from src.models.settings import ModelSettings
from src.runners.items import ModelResponse
from src.util.types import Response, ResponseOutput


@pytest.fixture
def mock_model_client():
    return AsyncMock()


@pytest.fixture
def model_responses(mock_model_client):
    return ModelResponsesModel("test-model", mock_model_client)


@pytest.fixture
def model_settings():
    return ModelSettings(
        temperature=0.7, top_p=0.9, max_tokens=100, sword_choice="auto", parallel_sword_calls=True
    )


@pytest.fixture
def mock_sword():
    sword = MagicMock(spec=Sword)
    sword.name = "test_sword"
    sword.description = "Test sword description"
    return sword


@pytest.fixture
def mock_orb():
    orb = MagicMock(spec=Orbs)
    orb.name = "test_orb"
    return orb


@pytest.mark.asyncio
async def test_get_response(model_responses, model_settings, mock_sword, mock_orb):
    # Setup
    system_instructions = "Test instructions"
    input_text = "Test input"
    output_schema = MagicMock(spec=AgentOutputSchema)

    # Mock the response
    mock_response = MagicMock(spec=Response)
    mock_response.output = [{"type": "text", "content": "Test response"}]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=20, total_tokens=30)
    mock_response.id = "test-id"
    model_responses._fetch_response = AsyncMock(return_value=mock_response)

    # Call the method
    result = await model_responses.get_response(
        system_instructions=system_instructions,
        input=input_text,
        model_settings=model_settings,
        swords=[mock_sword],
        output_schema=output_schema,
        orbs=[mock_orb],
    )

    assert isinstance(result, ModelResponse)
    assert result.output[0]["content"] == "Test response"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert result.usage.total_tokens == 30
    assert result.referenceable_id == "test-id"


@pytest.mark.asyncio
async def test_stream_response(model_responses, model_settings, mock_sword, mock_orb):
    # Setup
    system_instructions = "Test instructions"
    input_text = "Test input"
    output_schema = MagicMock(spec=AgentOutputSchema)

    # Mock the stream response
    async def mock_stream():
        yield MagicMock(spec=ResponseOutput, content="Chunk 1")
        yield MagicMock(spec=ResponseOutput, content="Chunk 2")

    model_responses._fetch_response = AsyncMock(return_value=mock_stream())

    # Call the method and collect results
    results = []
    async for chunk in model_responses.stream_response(
        system_instructions=system_instructions,
        input=input_text,
        model_settings=model_settings,
        swords=[mock_sword],
        output_schema=output_schema,
        orbs=[mock_orb],
    ):
        results.append(chunk)

    assert len(results) == 2
    assert results[0].content == "Chunk 1"
    assert results[1].content == "Chunk 2"


def test_converter_convert_sword_choice():
    # Test valid choices
    assert Converter.convert_sword_choice("auto") == "auto"
    assert Converter.convert_sword_choice("required") == "required"
    assert Converter.convert_sword_choice("none") == "none"
    # Test None input
    assert Converter.convert_sword_choice(None) == "auto"


def test_converter_get_response_format():
    # Test with output schema
    output_schema = MagicMock(spec=AgentOutputSchema)
    format = Converter.get_response_format(output_schema)
    assert format == {"type": "json_object"}


def test_converter_convert_swords(mock_sword, mock_orb):
    # Test conversion
    mock_sword.name = "test_sword"
    mock_sword.description = "Test sword description"
    mock_sword.params_json_schema = {"type": "object", "properties": {}}
    mock_orb.name = "test_orb"
    mock_orb.sword_name = "test_orb_sword"
    mock_orb.sword_description = "Test orb description"
    mock_orb.params_json_schema = {"type": "object", "properties": {}}

    result = Converter.convert_swords([mock_sword], [mock_orb])

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["name"] == "test_sword"
    assert result[0]["description"] == "Test sword description"
    assert result[0]["parameters"] == {"type": "object", "properties": {}}
    assert result[1]["name"] == "test_orb_sword"
    assert result[1]["description"] == "Test orb description"
    assert result[1]["parameters"] == {"type": "object", "properties": {}}
