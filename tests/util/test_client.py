from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.util._client import DeepSeekClient, setup_client


@pytest.fixture
def mock_http_client():
    return AsyncMock()


@pytest.fixture
def client(mock_http_client):
    return DeepSeekClient(
        api_key="test_key",
        base_url="http://test.com",
        organization="test_org",
        project="test_project",
        http_client=mock_http_client,
    )


def test_client_initialization():
    """Test DeepSeekClient initialization with default values."""
    client = DeepSeekClient()
    assert client.api_key == "API_KEY"
    assert client.base_url == "http://localhost:11434"
    assert client.organization is None
    assert client.project is None
    assert client.http_client is not None


def test_client_initialization_with_custom_values():
    """Test DeepSeekClient initialization with custom values."""
    client = DeepSeekClient(
        api_key="custom_key",
        base_url="http://custom.com",
        organization="custom_org",
        project="custom_project",
    )
    assert client.api_key == "custom_key"
    assert client.base_url == "http://custom.com"
    assert client.organization == "custom_org"
    assert client.project == "custom_project"


@pytest.mark.asyncio
async def test_chat_completion(client, mock_http_client):
    """Test chat completion creation."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_http_client.post.return_value = mock_response

    messages = [{"role": "user", "content": "Hello"}]
    result = await client.chat.completions.create(
        model="test-model",
        messages=messages,
        temperature=0.7,
        max_tokens=100,
    )

    assert isinstance(result, dict)
    assert result["model"] == "test-model"
    assert len(result["choices"]) == 1
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert result["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_chat_completion_with_streaming(client, mock_http_client):
    """Test chat completion with streaming enabled."""
    mock_response = MagicMock()

    # Create an async iterator from the list
    async def mock_aiter():
        yield 'data: {"choices": [{"delta": {"content": "Hello!"}}]}'

    mock_response.aiter_lines.return_value = mock_aiter()
    mock_http_client.post.return_value = mock_response

    messages = [{"role": "user", "content": "Hello"}]
    result = await client.chat.completions.create(
        model="test-model",
        messages=messages,
        stream=True,
    )

    assert hasattr(result, "__aiter__")
    async for chunk in result:
        assert "Hello!" in chunk["choices"][0]["delta"]["content"]


def test_setup_client():
    """Test setup_client function."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.timeout = MagicMock(connect=30.0, read=90.0)
        mock_client.return_value.limits = MagicMock(max_keepalive_connections=5, max_connections=10)
        client = setup_client()
        assert isinstance(client, DeepSeekClient)
        assert client.http_client.timeout.connect == 30.0
        assert client.http_client.timeout.read == 90.0
        assert client.http_client.limits.max_keepalive_connections == 5
        assert client.http_client.limits.max_connections == 10
