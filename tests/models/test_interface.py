from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.output import AgentOutputSchema
from src.gear.orbs import Orbs
from src.gear.sword import Sword
from src.models.interface import Model, ModelProvider
from src.models.settings import ModelSettings
from src.util._items import ModelResponse, TResponseInputItem
from src.util._types import ResponseStreamEvent, Usage


class MockModel(Model):
    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        swords: list[Sword],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
    ) -> ModelResponse:
        return ModelResponse(
            output=[{"type": "message", "content": "Mock response"}],
            usage=Usage(requests=1, input_tokens=0, output_tokens=0, total_tokens=0),
            referenceable_id=None,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        swords: list[Sword],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
    ) -> AsyncIterator[ResponseStreamEvent]:
        yield ResponseStreamEvent(type="message", content="Stream chunk 1")
        yield ResponseStreamEvent(type="message", content="Stream chunk 2")


class MockModelProvider(ModelProvider):
    def __init__(self):
        self.model = MockModel()

    def get_model(self, model_name: str | None) -> Model:
        return self.model


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_model_provider():
    return MockModelProvider()


@pytest.fixture
def model_settings():
    return ModelSettings(temperature=0.7, top_p=0.9, max_tokens=100)


@pytest.fixture
def mock_sword():
    sword = MagicMock(spec=Sword)
    sword.name = "test_sword"
    return sword


@pytest.fixture
def mock_orb():
    orb = MagicMock(spec=Orbs)
    orb.name = "test_orb"
    return orb


@pytest.mark.asyncio
async def test_model_get_response(mock_model, model_settings, mock_sword, mock_orb):
    system_instructions = "Test instructions"
    input_text = "Test input"
    output_schema = MagicMock(spec=AgentOutputSchema)

    response = await mock_model.get_response(
        system_instructions=system_instructions,
        input=input_text,
        model_settings=model_settings,
        swords=[mock_sword],
        output_schema=output_schema,
        orbs=[mock_orb],
    )

    assert isinstance(response, ModelResponse)
    assert response.output[0]["content"] == "Mock response"


@pytest.mark.asyncio
async def test_model_stream_response(mock_model, model_settings, mock_sword, mock_orb):
    system_instructions = "Test instructions"
    input_text = "Test input"
    output_schema = MagicMock(spec=AgentOutputSchema)

    async def mock_stream():
        yield {"choices": [{"delta": {"content": "Stream chunk 1"}}]}
        yield {"choices": [{"delta": {"content": "Stream chunk 2"}}]}

    mock_model.stream_response = AsyncMock(return_value=mock_stream())

    chunks = []
    stream = await mock_model.stream_response(
        system_instructions=system_instructions,
        input=input_text,
        model_settings=model_settings,
        swords=[mock_sword],
        output_schema=output_schema,
        orbs=[mock_orb],
    )

    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "Stream chunk 1"
    assert chunks[1]["choices"][0]["delta"]["content"] == "Stream chunk 2"


def test_model_provider_get_model(mock_model_provider):
    model = mock_model_provider.get_model("test-model")
    assert isinstance(model, Model)

    # Test that the model implements the required methods
    assert hasattr(model, "get_response")
    assert hasattr(model, "stream_response")

    # Test that the methods are async
    assert callable(model.get_response)
    assert callable(model.stream_response)
