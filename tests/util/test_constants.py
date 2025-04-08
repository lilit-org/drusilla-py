import logging
import os
from unittest.mock import patch

import pytest


def test_constants():
    """Test basic constants."""
    from src.util.constants import FAKE_RESPONSES_ID, THINK_TAGS, UNSET

    assert UNSET is not None
    assert FAKE_RESPONSES_ID == "fake_responses"
    assert isinstance(THINK_TAGS, tuple)
    assert len(THINK_TAGS) == 2
    assert THINK_TAGS[0] == "<think>"
    assert THINK_TAGS[1] == "</think>"


def test_supported_languages():
    """Test supported languages set."""
    from src.util.constants import SUPPORTED_LANGUAGES

    assert isinstance(SUPPORTED_LANGUAGES, set)
    assert len(SUPPORTED_LANGUAGES) > 0
    assert "en" in SUPPORTED_LANGUAGES  # English
    assert "es" in SUPPORTED_LANGUAGES  # Spanish
    assert "fr" in SUPPORTED_LANGUAGES  # French
    assert "de" in SUPPORTED_LANGUAGES  # German
    assert "zh" in SUPPORTED_LANGUAGES  # Chinese
    assert "ja" in SUPPORTED_LANGUAGES  # Japanese


def test_logging_config():
    """Test logging configuration."""
    from src.util.constants import LOG_LEVEL

    assert LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    logger = logging.getLogger("deepseek.agents")
    assert logger.level == getattr(logging, LOG_LEVEL)
    assert len(logger.handlers) > 0
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_connection_constants():
    """Test connection-related constants."""
    from src.util.constants import API_KEY, BASE_URL, MODEL

    assert isinstance(BASE_URL, str)
    assert BASE_URL.startswith(("http://", "https://"))
    assert isinstance(API_KEY, str)
    assert isinstance(MODEL, str)
    assert MODEL


def test_model_logic_constants():
    """Test model logic and optimization constants."""
    from src.util.constants import (
        LRU_CACHE_SIZE,
        MAX_GUARDRAIL_QUEUE_SIZE,
        MAX_QUEUE_SIZE,
        MAX_SHIELD_QUEUE_SIZE,
        MAX_TURNS,
    )

    assert isinstance(MAX_TURNS, int)
    assert MAX_TURNS > 0
    assert isinstance(MAX_QUEUE_SIZE, int)
    assert MAX_QUEUE_SIZE > 0
    assert isinstance(MAX_GUARDRAIL_QUEUE_SIZE, int)
    assert MAX_GUARDRAIL_QUEUE_SIZE > 0
    assert isinstance(MAX_SHIELD_QUEUE_SIZE, int)
    assert MAX_SHIELD_QUEUE_SIZE > 0
    assert isinstance(LRU_CACHE_SIZE, int)
    assert LRU_CACHE_SIZE > 0


def test_http_constants():
    """Test HTTP client configuration constants."""
    from src.util.constants import (
        HTTP_MAX_CONNECTIONS,
        HTTP_MAX_KEEPALIVE_CONNECTIONS,
        HTTP_TIMEOUT_CONNECT,
        HTTP_TIMEOUT_READ,
        HTTP_TIMEOUT_TOTAL,
    )

    assert isinstance(HTTP_TIMEOUT_TOTAL, float)
    assert HTTP_TIMEOUT_TOTAL > 0
    assert isinstance(HTTP_TIMEOUT_CONNECT, float)
    assert HTTP_TIMEOUT_CONNECT > 0
    assert isinstance(HTTP_TIMEOUT_READ, float)
    assert HTTP_TIMEOUT_READ > 0
    assert isinstance(HTTP_MAX_KEEPALIVE_CONNECTIONS, int)
    assert HTTP_MAX_KEEPALIVE_CONNECTIONS > 0
    assert isinstance(HTTP_MAX_CONNECTIONS, int)
    assert HTTP_MAX_CONNECTIONS > 0


def test_api_constants():
    """Test API configuration constants."""
    from src.util.constants import CHAT_COMPLETIONS_ENDPOINT, HEADERS

    assert isinstance(HEADERS, dict)
    assert "User-Agent" in HEADERS
    assert HEADERS["User-Agent"] == "Agents/Python"
    assert isinstance(CHAT_COMPLETIONS_ENDPOINT, str)
    assert CHAT_COMPLETIONS_ENDPOINT.startswith("/")


def test_get_env_var():
    """Test the get_env_var helper function."""
    from src.util.constants import get_env_var

    # Test with string default
    assert get_env_var("NONEXISTENT", "default") == "default"

    # Test with int default
    assert get_env_var("NONEXISTENT", 42, int) == 42

    # Test with float default
    assert get_env_var("NONEXISTENT", 3.14, float) == 3.14

    # Test with environment variable
    with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
        assert get_env_var("TEST_VAR", "default") == "test_value"


def test_config_class():
    """Test the Config class functionality."""
    from src.util.constants import Config

    config = Config()

    # Test default values
    assert isinstance(config.LOG_LEVEL, str)
    assert isinstance(config.BASE_URL, str)
    assert isinstance(config.API_KEY, str)
    assert isinstance(config.MODEL, str)
    assert isinstance(config.MAX_TURNS, int)
    assert isinstance(config.MAX_QUEUE_SIZE, int)
    assert isinstance(config.MAX_GUARDRAIL_QUEUE_SIZE, int)
    assert isinstance(config.MAX_SHIELD_QUEUE_SIZE, int)
    assert isinstance(config.LRU_CACHE_SIZE, int)
    assert isinstance(config.HTTP_TIMEOUT_TOTAL, float)
    assert isinstance(config.HTTP_TIMEOUT_CONNECT, float)
    assert isinstance(config.HTTP_TIMEOUT_READ, float)
    assert isinstance(config.HTTP_MAX_KEEPALIVE_CONNECTIONS, int)
    assert isinstance(config.HTTP_MAX_CONNECTIONS, int)

    # Test update_from_env
    with patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "INFO",
            "BASE_URL": "https://test.example.com",
            "API_KEY": "test_key",
            "MODEL": "test_model",
            "MAX_TURNS": "5",
            "MAX_QUEUE_SIZE": "500",
            "MAX_GUARDRAIL_QUEUE_SIZE": "50",
            "MAX_SHIELD_QUEUE_SIZE": "100",
            "LRU_CACHE_SIZE": "64",
            "HTTP_TIMEOUT_TOTAL": "60.0",
            "HTTP_TIMEOUT_CONNECT": "15.0",
            "HTTP_TIMEOUT_READ": "45.0",
            "HTTP_MAX_KEEPALIVE_CONNECTIONS": "3",
            "HTTP_MAX_CONNECTIONS": "5",
        },
    ):
        config.update_from_env()
        assert config.LOG_LEVEL == "INFO"
        assert config.BASE_URL == "https://test.example.com"
        assert config.API_KEY == "test_key"
        assert config.MODEL == "test_model"
        assert config.MAX_TURNS == 5
        assert config.MAX_QUEUE_SIZE == 500
        assert config.MAX_GUARDRAIL_QUEUE_SIZE == 50
        assert config.MAX_SHIELD_QUEUE_SIZE == 100
        assert config.LRU_CACHE_SIZE == 64
        assert config.HTTP_TIMEOUT_TOTAL == 60.0
        assert config.HTTP_TIMEOUT_CONNECT == 15.0
        assert config.HTTP_TIMEOUT_READ == 45.0
        assert config.HTTP_MAX_KEEPALIVE_CONNECTIONS == 3
        assert config.HTTP_MAX_CONNECTIONS == 5


def test_error_message_class():
    """Test the ErrorMessage class."""
    from src.util.constants import ErrorMessage

    error = ErrorMessage(message="Test error", used_in="test.py")
    assert error.message == "Test error"
    assert error.used_in == "test.py"


def test_error_messages_class():
    """Test the ErrorMessages class."""
    from src.util.constants import ErrorMessage, ErrorMessages

    # Clear any existing error messages
    ErrorMessages._error_messages.clear()

    # Test with no environment variables
    with patch.dict(os.environ, {}, clear=True):
        error_messages = ErrorMessages()
        assert error_messages._error_messages == {}

    # Test with environment variables
    with patch.dict(
        os.environ,
        {
            "SWORD_ERROR_MESSAGE": "Sword error",
            "RUNCONTEXT_ERROR_MESSAGE": "Context error",
            "SHIELD_ERROR_MESSAGE": "Shield error",
            "RUNNER_ERROR_MESSAGE": "Runner error",
            "ORBS_ERROR_MESSAGE": "Orbs error",
            "AGENT_EXEC_ERROR_MESSAGE": "Agent error",
        },
    ):
        error_messages = ErrorMessages()
        assert isinstance(error_messages.SWORD_ERROR, ErrorMessage)
        assert error_messages.SWORD_ERROR.message == "Sword error"
        assert error_messages.SWORD_ERROR.used_in == "src/gear/sword.py"

        # Test __getattr__ with non-existent error
        with pytest.raises(AttributeError) as exc_info:
            _ = error_messages.NONEXISTENT_ERROR
        assert "NONEXISTENT_ERROR" in str(exc_info.value)


def test_validate_required_env_vars():
    """Test the validate_required_env_vars function."""
    from src.util.constants import validate_required_env_vars

    # Test with all required variables
    with patch.dict(
        os.environ,
        {
            "SWORD_ERROR_MESSAGE": "Sword error",
            "RUNCONTEXT_ERROR_MESSAGE": "Context error",
            "SHIELD_ERROR_MESSAGE": "Shield error",
            "RUNNER_ERROR_MESSAGE": "Runner error",
            "ORBS_ERROR_MESSAGE": "Orbs error",
            "AGENT_EXEC_ERROR_MESSAGE": "Agent error",
        },
    ):
        validate_required_env_vars()  # Should not raise

    # Test with missing variables
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            validate_required_env_vars()
        assert "Missing required environment variables" in str(exc_info.value)
        missing_vars = [
            "SWORD_ERROR_MESSAGE",
            "RUNCONTEXT_ERROR_MESSAGE",
            "SHIELD_ERROR_MESSAGE",
            "RUNNER_ERROR_MESSAGE",
            "ORBS_ERROR_MESSAGE",
            "AGENT_EXEC_ERROR_MESSAGE",
        ]
        for var in missing_vars:
            assert var in str(exc_info.value)


def test_load_environment():
    """Test the load_environment function."""
    from src.util.constants import load_environment, logger

    # Save original environment
    original_env = dict(os.environ)

    try:
        # Set test environment variables
        with patch.dict(
            os.environ,
            {
                "LOG_LEVEL": "INFO",
                "BASE_URL": "https://test.example.com",
                "API_KEY": "test_key",
                "MODEL": "test_model",
                "MAX_TURNS": "5",
                "MAX_QUEUE_SIZE": "500",
                "MAX_GUARDRAIL_QUEUE_SIZE": "50",
                "MAX_SHIELD_QUEUE_SIZE": "100",
                "LRU_CACHE_SIZE": "64",
                "HTTP_TIMEOUT_TOTAL": "60.0",
                "HTTP_TIMEOUT_CONNECT": "15.0",
                "HTTP_TIMEOUT_READ": "45.0",
                "HTTP_MAX_KEEPALIVE_CONNECTIONS": "3",
                "HTTP_MAX_CONNECTIONS": "5",
                "SWORD_ERROR_MESSAGE": "Sword error",
                "RUNCONTEXT_ERROR_MESSAGE": "Context error",
                "SHIELD_ERROR_MESSAGE": "Shield error",
                "RUNNER_ERROR_MESSAGE": "Runner error",
                "ORBS_ERROR_MESSAGE": "Orbs error",
                "AGENT_EXEC_ERROR_MESSAGE": "Agent error",
            },
        ):
            # Patch load_dotenv to do nothing
            with patch("src.util.constants.load_dotenv"):
                load_environment()

                # Verify logger configuration
                assert logger.level == logging.INFO

                # Verify global variables
                from src.util.constants import (
                    API_KEY,
                    BASE_URL,
                    HTTP_MAX_CONNECTIONS,
                    HTTP_MAX_KEEPALIVE_CONNECTIONS,
                    HTTP_TIMEOUT_CONNECT,
                    HTTP_TIMEOUT_READ,
                    HTTP_TIMEOUT_TOTAL,
                    LOG_LEVEL,
                    LRU_CACHE_SIZE,
                    MAX_GUARDRAIL_QUEUE_SIZE,
                    MAX_QUEUE_SIZE,
                    MAX_SHIELD_QUEUE_SIZE,
                    MAX_TURNS,
                    MODEL,
                )

                assert LOG_LEVEL == "INFO"
                assert BASE_URL == "https://test.example.com"
                assert API_KEY == "test_key"
                assert MODEL == "test_model"
                assert MAX_TURNS == 5
                assert MAX_QUEUE_SIZE == 500
                assert MAX_GUARDRAIL_QUEUE_SIZE == 50
                assert MAX_SHIELD_QUEUE_SIZE == 100
                assert LRU_CACHE_SIZE == 64
                assert HTTP_TIMEOUT_TOTAL == 60.0
                assert HTTP_TIMEOUT_CONNECT == 15.0
                assert HTTP_TIMEOUT_READ == 45.0
                assert HTTP_MAX_KEEPALIVE_CONNECTIONS == 3
                assert HTTP_MAX_CONNECTIONS == 5
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)
