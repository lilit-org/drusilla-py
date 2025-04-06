from unittest.mock import AsyncMock

import pytest

from src.models.shared import (
    SharedConfig,
    get_default_model_client,
    get_default_model_key,
    get_use_responses_by_default,
    set_default_model_client,
    set_default_model_key,
    set_use_responses_by_default,
)
from src.util._types import AsyncDeepSeek


@pytest.fixture
def mock_client():
    return AsyncMock(spec=AsyncDeepSeek)


@pytest.fixture(autouse=True)
def reset_shared_config():
    """Reset the shared config before each test"""
    SharedConfig._instance = None
    yield
    SharedConfig._instance = None


def test_shared_config_singleton():
    # Get first instance
    instance1 = SharedConfig.get_instance()
    assert isinstance(instance1, SharedConfig)

    # Get second instance - should be the same
    instance2 = SharedConfig.get_instance()
    assert instance2 is instance1

    # Reset and get new instance
    SharedConfig._instance = None
    instance3 = SharedConfig.get_instance()
    assert instance3 is not instance1


def test_shared_config_initialization():
    config = SharedConfig.get_instance()

    assert config.model_key is None
    assert config.model_client is None
    assert config.use_responses is False


def test_set_get_default_model_key():
    # Test setting key
    set_default_model_key("test-key")
    assert get_default_model_key() == "test-key"

    # Test setting None
    set_default_model_key(None)
    assert get_default_model_key() is None


def test_set_get_default_model_client(mock_client):
    # Test setting client
    set_default_model_client(mock_client)
    assert get_default_model_client() is mock_client

    # Test setting None
    set_default_model_client(None)
    assert get_default_model_client() is None


def test_set_get_use_responses_by_default():
    # Test setting to True
    set_use_responses_by_default(True)
    assert get_use_responses_by_default() is True

    # Test setting to False
    set_use_responses_by_default(False)
    assert get_use_responses_by_default() is False


def test_shared_config_persistence():
    # Set values
    set_default_model_key("test-key")
    set_use_responses_by_default(True)

    # Get new instance - values should persist
    config = SharedConfig.get_instance()
    assert config.model_key == "test-key"
    assert config.use_responses is True


def test_shared_config_reset():
    # Set values
    set_default_model_key("test-key")
    set_use_responses_by_default(True)

    # Reset instance
    SharedConfig._instance = None

    # Get new instance - values should be default
    config = SharedConfig.get_instance()
    assert config.model_key is None
    assert config.use_responses is False


def test_shared_config_thread_safety():
    # This test is more of a documentation of the potential issue
    # since we can't easily test thread safety in a simple test
    config1 = SharedConfig.get_instance()
    config2 = SharedConfig.get_instance()

    # In a multi-threaded environment, these could be different
    # if the singleton pattern isn't properly synchronized
    assert config1 is config2
