import logging
import os
from unittest.mock import patch


def test_constants():
    """Test basic constants."""
    from src.util._constants import FAKE_RESPONSES_ID, UNSET

    assert UNSET is not None
    assert FAKE_RESPONSES_ID == "fake_responses"


def test_supported_languages():
    """Test supported languages set."""
    from src.util._constants import SUPPORTED_LANGUAGES

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
    from src.util._constants import LOG_LEVEL

    assert LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    logger = logging.getLogger("deepseek.agents")
    assert logger.level == getattr(logging, LOG_LEVEL)
    assert len(logger.handlers) > 0
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_connection_constants():
    """Test connection-related constants."""
    from src.util._constants import API_KEY, BASE_URL, MODEL

    assert isinstance(BASE_URL, str)
    assert BASE_URL.startswith(("http://", "https://"))
    assert isinstance(API_KEY, str)
    assert isinstance(MODEL, str)
    assert MODEL


def test_model_logic_constants():
    """Test model logic and optimization constants."""
    from src.util._constants import (
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
    from src.util._constants import (
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
    from src.util._constants import CHAT_COMPLETIONS_ENDPOINT, HEADERS

    assert isinstance(HEADERS, dict)
    assert "User-Agent" in HEADERS
    assert HEADERS["User-Agent"] == "Agents/Python"
    assert isinstance(CHAT_COMPLETIONS_ENDPOINT, str)
    assert CHAT_COMPLETIONS_ENDPOINT.startswith("/")


def test_environment_variables():
    """Test that environment variables are properly loaded."""
    # Save original environment variables
    original_env = {
        "LOG_LEVEL": os.environ.get("LOG_LEVEL"),
        "BASE_URL": os.environ.get("BASE_URL"),
        "API_KEY": os.environ.get("API_KEY"),
        "MODEL": os.environ.get("MODEL"),
        "MAX_TURNS": os.environ.get("MAX_TURNS"),
        "MAX_QUEUE_SIZE": os.environ.get("MAX_QUEUE_SIZE"),
        "MAX_GUARDRAIL_QUEUE_SIZE": os.environ.get("MAX_GUARDRAIL_QUEUE_SIZE"),
        "LRU_CACHE_SIZE": os.environ.get("LRU_CACHE_SIZE"),
        "HTTP_TIMEOUT_TOTAL": os.environ.get("HTTP_TIMEOUT_TOTAL"),
        "HTTP_TIMEOUT_CONNECT": os.environ.get("HTTP_TIMEOUT_CONNECT"),
        "HTTP_TIMEOUT_READ": os.environ.get("HTTP_TIMEOUT_READ"),
        "HTTP_MAX_KEEPALIVE_CONNECTIONS": os.environ.get("HTTP_MAX_KEEPALIVE_CONNECTIONS"),
        "HTTP_MAX_CONNECTIONS": os.environ.get("HTTP_MAX_CONNECTIONS"),
    }

    try:
        # Set test environment variables
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["BASE_URL"] = "https://test.example.com"
        os.environ["API_KEY"] = "test_key"
        os.environ["MODEL"] = "test_model"
        os.environ["MAX_TURNS"] = "5"
        os.environ["MAX_QUEUE_SIZE"] = "500"
        os.environ["MAX_GUARDRAIL_QUEUE_SIZE"] = "50"
        os.environ["LRU_CACHE_SIZE"] = "64"
        os.environ["HTTP_TIMEOUT_TOTAL"] = "60.0"
        os.environ["HTTP_TIMEOUT_CONNECT"] = "15.0"
        os.environ["HTTP_TIMEOUT_READ"] = "45.0"
        os.environ["HTTP_MAX_KEEPALIVE_CONNECTIONS"] = "3"
        os.environ["HTTP_MAX_CONNECTIONS"] = "5"

        # Import the module after setting environment variables
        import importlib

        from src.util import _constants

        importlib.reload(_constants)

        # Patch load_dotenv to do nothing
        with patch("src.util._constants.load_dotenv"):
            # Reload environment variables
            _constants.load_environment()

            assert _constants.LOG_LEVEL == "DEBUG"
            assert _constants.BASE_URL == "https://test.example.com"
            assert _constants.API_KEY == "test_key"
            assert _constants.MODEL == "test_model"
            assert _constants.MAX_TURNS == 5
            assert _constants.MAX_QUEUE_SIZE == 500
            assert _constants.MAX_GUARDRAIL_QUEUE_SIZE == 50
            assert _constants.LRU_CACHE_SIZE == 64
            assert _constants.HTTP_TIMEOUT_TOTAL == 60.0
            assert _constants.HTTP_TIMEOUT_CONNECT == 15.0
            assert _constants.HTTP_TIMEOUT_READ == 45.0
            assert _constants.HTTP_MAX_KEEPALIVE_CONNECTIONS == 3
            assert _constants.HTTP_MAX_CONNECTIONS == 5

    finally:
        # Restore original environment variables
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
