from unittest.mock import ANY, AsyncMock, patch

import pytest

from src.models.chat import ModelChatCompletionsModel
from src.models.provider import ModelProvider
from src.models.responses import ModelResponsesModel
from src.util._types import AsyncDeepSeek


@pytest.fixture
def mock_client():
    return AsyncMock(spec=AsyncDeepSeek)


@pytest.fixture
def provider(mock_client):
    return ModelProvider(
        api_key="test-api-key",
        base_url="https://test-api.com",
        organization="test-org",
        project="test-project",
        use_responses=False,
    )


def test_provider_initialization():
    # Test with all parameters
    provider = ModelProvider(
        api_key="test-api-key",
        base_url="https://test-api.com",
        organization="test-org",
        project="test-project",
        use_responses=True,
    )

    assert provider._stored_api_key == "test-api-key"
    assert provider._stored_base_url == "https://test-api.com"
    assert provider._stored_organization == "test-org"
    assert provider._stored_project == "test-project"
    assert provider._use_responses is True


def test_provider_initialization_defaults():
    # Test with minimal parameters
    provider = ModelProvider()

    assert provider._stored_api_key is None
    assert provider._stored_base_url is not None
    assert provider._stored_organization is None
    assert provider._stored_project is None
    assert provider._use_responses is False


def test_provider_get_client(provider, mock_client):
    # Test client initialization
    with patch("src.models.provider.DeepSeekClient") as mock_client_class:
        mock_client_class.return_value = mock_client
        client = provider._get_client()

        assert client == mock_client
        mock_client_class.assert_called_once_with(
            api_key="test-api-key",
            base_url="https://test-api.com",
            organization="test-org",
            project="test-project",
            http_client=ANY,
        )


def test_provider_get_model_chat(provider, mock_client):
    # Test getting chat model
    provider._client = mock_client
    model = provider.get_model("test-model")

    assert isinstance(model, ModelChatCompletionsModel)
    assert model.model == "test-model"
    assert model._client == mock_client


def test_provider_get_model_responses(provider, mock_client):
    # Test getting responses model
    provider.use_responses = True
    provider._client = mock_client
    model = provider.get_model("test-model")

    assert isinstance(model, ModelResponsesModel)
    assert model.model == "test-model"
    assert model._client == mock_client


def test_provider_get_model_default(provider, mock_client):
    # Test getting model with default name
    provider._client = mock_client
    model = provider.get_model()

    assert isinstance(model, ModelChatCompletionsModel)
    assert model.model == "deepseek-r1"  # Default model name
    assert model._client == mock_client


def test_provider_get_model_lazy_client():
    # Test that client is initialized when getting model
    provider = ModelProvider(api_key="test-api-key")

    with patch("src.models.provider.DeepSeekClient") as mock_client_class:
        mock_client = AsyncMock(spec=AsyncDeepSeek)
        mock_client_class.return_value = mock_client

        model = provider.get_model("test-model")

        assert provider._client == mock_client
        assert model._client == mock_client
