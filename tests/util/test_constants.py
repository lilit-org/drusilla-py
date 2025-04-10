import logging
import os
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from src.util.constants import BaseConfig, Config, config, HEADERS, logger


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Setup environment variables for tests."""
    # Clear any existing environment variables
    for key in [
        "THINK_TAGS",
        "SUPPORTED_LANGUAGES",
        "API_KEY",
        "LOG_LEVEL",
        "BASE_URL",
        "MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)

    # Set up clean environment with properly escaped values
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("BASE_URL", "http://test.com")
    monkeypatch.setenv("MODEL", "test-model")

    # Import after environment is set
    global config
    config = Config()
    yield config


def test_logging_config():
    """Test logging configuration."""
    assert config.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    assert logger.level == getattr(logging, config.LOG_LEVEL)
    assert len(logger.handlers) > 0
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_connection_constants():
    """Test connection-related constants."""
    assert isinstance(config.BASE_URL, str)
    assert config.BASE_URL.startswith(("http://", "https://"))
    assert isinstance(config.API_KEY, str)
    assert isinstance(config.MODEL, str)
    assert config.MODEL


def test_model_logic_constants():
    """Test model logic and optimization constants."""
    assert isinstance(config.MAX_TURNS, int)
    assert config.MAX_TURNS > 0
    assert isinstance(config.MAX_QUEUE_SIZE, int)
    assert config.MAX_QUEUE_SIZE > 0
    assert isinstance(config.MAX_GUARDRAIL_QUEUE_SIZE, int)
    assert config.MAX_GUARDRAIL_QUEUE_SIZE > 0
    assert isinstance(config.MAX_SHIELD_QUEUE_SIZE, int)
    assert config.MAX_SHIELD_QUEUE_SIZE > 0
    assert isinstance(config.LRU_CACHE_SIZE, int)
    assert config.LRU_CACHE_SIZE > 0


def test_http_constants():
    """Test HTTP client configuration constants."""
    assert isinstance(config.HTTP_TIMEOUT_TOTAL, float)
    assert config.HTTP_TIMEOUT_TOTAL > 0
    assert isinstance(config.HTTP_TIMEOUT_CONNECT, float)
    assert config.HTTP_TIMEOUT_CONNECT > 0
    assert isinstance(config.HTTP_TIMEOUT_READ, float)
    assert config.HTTP_TIMEOUT_READ > 0
    assert isinstance(config.HTTP_MAX_KEEPALIVE_CONNECTIONS, int)
    assert config.HTTP_MAX_KEEPALIVE_CONNECTIONS > 0
    assert isinstance(config.HTTP_MAX_CONNECTIONS, int)
    assert config.HTTP_MAX_CONNECTIONS > 0


def test_api_constants():
    """Test API configuration constants."""
    assert isinstance(HEADERS, dict)
    assert "User-Agent" in HEADERS
    assert HEADERS["User-Agent"] == config.USER_AGENT
    assert isinstance(config.CHAT_COMPLETIONS_ENDPOINT, str)
    assert config.CHAT_COMPLETIONS_ENDPOINT.startswith("/")


def test_base_config():
    """Test the BaseConfig class functionality."""

    @dataclass
    class TestConfig(BaseConfig):
        TEST_STR: str = "default"
        TEST_INT: int = 42
        TEST_FLOAT: float = 3.14
        TEST_BOOL: bool = False

    config = TestConfig()

    # Test get_env_var with None values
    assert config.get_env_var("NONEXISTENT", None) is None
    assert config.get_env_var("NONEXISTENT", None, str) is None

    # Test get_env_var with invalid type conversions
    with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
        assert config.get_env_var("TEST_INT", 42, int) == 42
    with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
        assert config.get_env_var("TEST_FLOAT", 3.14, float) == 3.14

    # Test get_env_var with valid values
    assert config.get_env_var("NONEXISTENT", "default") == "default"
    assert config.get_env_var("NONEXISTENT", 42, int) == 42
    assert config.get_env_var("NONEXISTENT", 3.14, float) == 3.14

    # Test boolean values
    assert config.get_env_var("NONEXISTENT", True, bool) is True
    with patch.dict(os.environ, {"TEST_BOOL": "true"}):
        assert config.get_env_var("TEST_BOOL", False, bool) is True
    with patch.dict(os.environ, {"TEST_BOOL": "1"}):
        assert config.get_env_var("TEST_BOOL", False, bool) is True
    with patch.dict(os.environ, {"TEST_BOOL": "false"}):
        assert config.get_env_var("TEST_BOOL", True, bool) is False
    with patch.dict(os.environ, {"TEST_BOOL": "0"}):
        assert config.get_env_var("TEST_BOOL", True, bool) is False
    with patch.dict(os.environ, {"TEST_BOOL": "invalid"}):
        assert config.get_env_var("TEST_BOOL", True, bool) is True

    # Test update_from_env with invalid values
    with patch.dict(
        os.environ,
        {
            "TEST_STR": "new_value",
            "TEST_INT": "not_a_number",
            "TEST_FLOAT": "not_a_float",
            "TEST_BOOL": "invalid_bool",
        },
    ):
        config.update_from_env()
        assert config.TEST_STR == "new_value"
        assert config.TEST_INT == 42  # Should keep default for invalid int
        assert config.TEST_FLOAT == 3.14  # Should keep default for invalid float
        assert config.TEST_BOOL is False  # Should keep default for invalid bool

    # Test update_from_env with valid values
    with patch.dict(
        os.environ,
        {
            "TEST_STR": "new_value",
            "TEST_INT": "100",
            "TEST_FLOAT": "6.28",
            "TEST_BOOL": "true",
        },
    ):
        config.update_from_env()
        assert config.TEST_STR == "new_value"
        assert config.TEST_INT == 100
        assert config.TEST_FLOAT == 6.28
        assert config.TEST_BOOL is True


def test_error_messages():
    """Test error message constants."""
    from src.util.constants import err

    # Test that all error messages are properly formatted
    test_error = "test error"
    assert err.SWORD_ERROR.format(error=test_error) == f"Sword error: {test_error}"
    assert err.SHIELD_ERROR.format(error=test_error) == f"Shield error: {test_error}"
    assert err.RUNCONTEXT_ERROR.format(error=test_error) == f"RunContextWrapper error: {test_error}"
    assert err.RUNNER_ERROR.format(error=test_error) == f"Runner error: {test_error}"
    assert err.ORBS_ERROR.format(error=test_error) == f"Orbs error: {test_error}"
    assert err.AGENT_EXEC_ERROR.format(error=test_error) == f"Agent execution error: {test_error}"
    assert err.MODEL_ERROR.format(error=test_error) == f"Model error: {test_error}"
    assert err.TYPES_ERROR.format(error=test_error) == f"Type error: {test_error}"
    assert err.OBJECT_ADDITIONAL_PROPERTIES_ERROR == (
        "Object types cannot allow additional properties. This may be due to using an older "
        "Pydantic version or explicit configuration. If needed, update the function or output "
        "sword to use a non-strict schema."
    )


def test_validate_env_vars():
    """Test the validate_env_vars function."""
    from src.util.constants import BaseConfig, Config, ErrMsg

    # Test with all required variables
    with patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "DEBUG",
            "BASE_URL": "http://localhost:11434",
            "API_KEY": "",
            "MODEL": "deepseek-r1",
            "USER_AGENT": "Agents/Python",
            "MAX_TURNS": "10",
            "MAX_QUEUE_SIZE": "1000",
            "MAX_GUARDRAIL_QUEUE_SIZE": "100",
            "MAX_SHIELD_QUEUE_SIZE": "1000",
            "LRU_CACHE_SIZE": "128",
            "HTTP_TIMEOUT_TOTAL": "120.0",
            "HTTP_TIMEOUT_CONNECT": "30.0",
            "HTTP_TIMEOUT_READ": "90.0",
            "HTTP_MAX_KEEPALIVE_CONNECTIONS": "5",
            "HTTP_MAX_CONNECTIONS": "10",
            "CHAT_COMPLETIONS_ENDPOINT": "/api/chat",
            "THINK_TAGS": "('<think>', '</think>')",
            "SUPPORTED_LANGUAGES": (
                "{'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', "
                "'zh', 'ar', 'hi', 'nl', 'pl', 'tr', 'vi', 'he'}"
            ),
        },
        clear=True,
    ):
        # Should not raise any errors
        BaseConfig.validate_all(Config, ErrMsg)

    # Test with missing required variables
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            BaseConfig.validate_all(Config, ErrMsg)
        assert "Missing required environment variables" in str(exc_info.value)

    # Test with invalid values
    with patch.dict(
        os.environ,
        {
            "API_KEY": "test",
            "MAX_TURNS": "invalid",
            "HTTP_TIMEOUT_TOTAL": "not_a_float",
        },
        clear=True,
    ):
        with pytest.raises(ValueError) as exc_info:
            BaseConfig.validate_all(Config, ErrMsg)
        assert "Invalid environment variable values" in str(exc_info.value)


def test_convert_env_value_edge_cases():
    """Test edge cases for _convert_env_value method."""
    from src.util.constants import BaseConfig

    class TestConfig(BaseConfig):
        TEST_INT: int = 42
        TEST_FLOAT: float = 3.14
        TEST_BOOL: bool = False

    config = TestConfig()

    # Test invalid numeric conversion
    assert config._convert_env_value("not_a_number", int, 42) == 42
    assert config._convert_env_value("not_a_float", float, 3.14) == 3.14

    # Test invalid boolean conversion
    assert config._convert_env_value("invalid_bool", bool, False) is False


def test_error_messages_edge_cases():
    """Test edge cases for error messages."""
    from src.util.constants import ErrMsg

    # Test with empty environment
    error_msgs = ErrMsg()
    assert error_msgs.SWORD_ERROR == "Sword error: {error}"

    # Test with non-string error message
    with patch.dict(os.environ, {"SWORD_ERROR": "123"}):
        error_msgs = ErrMsg()
        assert error_msgs.SWORD_ERROR == "123"


def test_validate_all_edge_cases():
    """Test edge cases for validate_all method."""
    from src.util.constants import BaseConfig, Config, ErrMsg

    # Test with empty environment
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            BaseConfig.validate_all(Config, ErrMsg)
        assert "Missing required environment variables" in str(exc_info.value)

    # Test with invalid numeric values
    with patch.dict(
        os.environ,
        {
            "API_KEY": "test",
            "MAX_TURNS": "not_a_number",
            "HTTP_TIMEOUT_TOTAL": "not_a_float",
            "MAX_QUEUE_SIZE": "-1",
        },
        clear=True,
    ):
        with pytest.raises(ValueError) as exc_info:
            BaseConfig.validate_all(Config, ErrMsg)
        assert "Invalid environment variable values" in str(exc_info.value)


def test_get_env_var_edge_cases():
    """Test edge cases for get_env_var method."""
    from src.util.constants import BaseConfig

    # Test with None value
    assert BaseConfig.get_env_var("NONEXISTENT", None) is None

    # Test with empty string
    assert BaseConfig.get_env_var("NONEXISTENT", "") == ""

    # Test with invalid type conversion
    assert BaseConfig.get_env_var("NONEXISTENT", 42, int) == 42
    assert BaseConfig.get_env_var("NONEXISTENT", 3.14, float) == 3.14

    # Test with invalid boolean values
    assert BaseConfig.get_env_var("NONEXISTENT", True, bool) is True
    with patch.dict(os.environ, {"TEST_BOOL": "invalid"}):
        assert BaseConfig.get_env_var("TEST_BOOL", True, bool) is True


def test_update_from_env_edge_cases():
    """Test edge cases for update_from_env method."""

    @dataclass
    class TestConfig(BaseConfig):
        TEST_STR: str = "default"
        TEST_INT: int = 42
        TEST_FLOAT: float = 3.14
        TEST_BOOL: bool = False

    config = TestConfig()

    # Test with empty environment
    config.update_from_env()
    assert config.TEST_STR == "default"
    assert config.TEST_INT == 42
    assert config.TEST_FLOAT == 3.14
    assert config.TEST_BOOL is False

    # Test with invalid values
    with patch.dict(
        os.environ,
        {
            "TEST_STR": "",
            "TEST_INT": "not_a_number",
            "TEST_FLOAT": "not_a_float",
            "TEST_BOOL": "invalid",
        },
    ):
        config.update_from_env()
        assert config.TEST_STR == ""
        assert config.TEST_INT == 42
        assert config.TEST_FLOAT == 3.14
        assert config.TEST_BOOL is False
