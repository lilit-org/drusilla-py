"""
This module contains API configurations, environment settings,
and default values.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import ClassVar

from dotenv import find_dotenv, load_dotenv

from .types import T

__all__ = (
    "HEADERS",
    "UNSET",
    "FAKE_RESPONSES_ID",
    "THINK_TAGS",
    "Config",
    "CHAT_COMPLETIONS_ENDPOINT",
    "SUPPORTED_LANGUAGES",
    "load_environment",
    "logger",
    "ERROR_MESSAGES",
)

# Load .env file
load_dotenv(find_dotenv())

# Core Constants
UNSET = object()
FAKE_RESPONSES_ID = "fake_responses"
THINK_TAGS = ("<think>", "</think>")

# API Configuration
_USER_AGENT = "Agents/Python"
HEADERS = {"User-Agent": _USER_AGENT}
CHAT_COMPLETIONS_ENDPOINT = "/api/chat"

# Supported languages for translation
SUPPORTED_LANGUAGES: set[str] = {
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "pt",  # Portuguese
    "ru",  # Russian
    "ja",  # Japanese
    "ko",  # Korean
    "zh",  # Chinese
    "ar",  # Arabic
    "hi",  # Hindi
    "nl",  # Dutch
    "pl",  # Polish
    "tr",  # Turkish
    "vi",  # Vietnamese
    "he",  # Hebrew
}

# Initialize logger
logger = logging.getLogger("deepseek.agents")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def get_env_var(name: str, default: T, type_func: type = str) -> T:
    """Helper function to get and type cast environment variables."""
    value = os.getenv(name, default)
    return type_func(value) if value is not None else default


@dataclass
class Config:
    """Configuration class to manage environment variables and their defaults."""

    LOG_LEVEL: str = field(default_factory=lambda: get_env_var("DEFAULT_LOG_LEVEL", "DEBUG"))
    BASE_URL: str = field(
        default_factory=lambda: get_env_var("DEFAULT_BASE_URL", "http://localhost:11434")
    )
    API_KEY: str = field(default_factory=lambda: get_env_var("DEFAULT_API_KEY", ""))
    MODEL: str = field(default_factory=lambda: get_env_var("DEFAULT_MODEL", "deepseek-r1"))
    MAX_TURNS: int = field(default_factory=lambda: get_env_var("DEFAULT_MAX_TURNS", 10, int))
    MAX_QUEUE_SIZE: int = field(
        default_factory=lambda: get_env_var("DEFAULT_MAX_QUEUE_SIZE", 1000, int)
    )
    MAX_GUARDRAIL_QUEUE_SIZE: int = field(
        default_factory=lambda: get_env_var("DEFAULT_MAX_GUARDRAIL_QUEUE_SIZE", 100, int)
    )
    MAX_SHIELD_QUEUE_SIZE: int = field(
        default_factory=lambda: get_env_var("DEFAULT_MAX_SHIELD_QUEUE_SIZE", 1000, int)
    )
    LRU_CACHE_SIZE: int = field(
        default_factory=lambda: get_env_var("DEFAULT_LRU_CACHE_SIZE", 128, int)
    )
    HTTP_TIMEOUT_TOTAL: float = field(
        default_factory=lambda: get_env_var("DEFAULT_HTTP_TIMEOUT_TOTAL", 120.0, float)
    )
    HTTP_TIMEOUT_CONNECT: float = field(
        default_factory=lambda: get_env_var("DEFAULT_HTTP_TIMEOUT_CONNECT", 30.0, float)
    )
    HTTP_TIMEOUT_READ: float = field(
        default_factory=lambda: get_env_var("DEFAULT_HTTP_TIMEOUT_READ", 90.0, float)
    )
    HTTP_MAX_KEEPALIVE_CONNECTIONS: int = field(
        default_factory=lambda: get_env_var("DEFAULT_HTTP_MAX_KEEPALIVE_CONNECTIONS", 5, int)
    )
    HTTP_MAX_CONNECTIONS: int = field(
        default_factory=lambda: get_env_var("DEFAULT_HTTP_MAX_CONNECTIONS", 10, int)
    )

    def update_from_env(self) -> None:
        """Update configuration values from environment variables."""
        for field_name in self.__dataclass_fields__:
            env_key = field_name
            if hasattr(self, field_name):
                current_value = getattr(self, field_name)
                new_value = get_env_var(env_key, current_value, type(current_value))
                setattr(self, field_name, new_value)


@dataclass(frozen=True)
class ErrorMessage:
    message: str
    used_in: str


@dataclass(frozen=True)
class ErrorMessages:
    """Collection of error messages used throughout the application."""

    _error_messages: ClassVar[dict[str, ErrorMessage]] = {}

    def __post_init__(self):
        """Initialize error messages from environment variables."""
        error_vars = {
            "SWORD_ERROR": ("SWORD_ERROR_MESSAGE", "src/gear/sword.py"),
            "RUNCONTEXT_ERROR": ("RUNCONTEXT_ERROR_MESSAGE", "src/gear/sword.py"),
            "SHIELD_ERROR": ("SHIELD_ERROR_MESSAGE", "src/gear/shield.py"),
            "RUNNER_ERROR": ("RUNNER_ERROR_MESSAGE", "src/runners/run.py"),
            "ORBS_ERROR": ("ORBS_ERROR_MESSAGE", "src/gear/orbs.py"),
            "AGENT_EXEC_ERROR": ("AGENT_EXEC_ERROR_MESSAGE", "src/agents/agent.py"),
            "MODEL_ERROR": ("MODEL_ERROR_MESSAGE", "src/util/print.py"),
            "TYPES_ERROR": ("TYPES_ERROR_MESSAGE", "src/util/types.py"),
            "OBJECT_ADDITIONAL_PROPERTIES_ERROR": (
                "OBJECT_ADDITIONAL_PROPERTIES_ERROR",
                "src/util/schema.py",
            ),
        }

        for name, (env_var, used_in) in error_vars.items():
            if env_var in os.environ:
                self._error_messages[name] = ErrorMessage(
                    message=os.environ[env_var], used_in=used_in
                )

    def __getattr__(self, name: str) -> ErrorMessage:
        """Get error message by name."""
        if name in self._error_messages:
            return self._error_messages[name]
        raise AttributeError(f"No error message defined for {name}")


def validate_required_env_vars() -> None:
    """Validate that all required environment variables are present."""
    required_vars = {
        "SWORD_ERROR_MESSAGE",
        "RUNCONTEXT_ERROR_MESSAGE",
        "SHIELD_ERROR_MESSAGE",
        "RUNNER_ERROR_MESSAGE",
        "ORBS_ERROR_MESSAGE",
        "AGENT_EXEC_ERROR_MESSAGE",
        "MODEL_ERROR_MESSAGE",
        "TYPES_ERROR_MESSAGE",
        "OBJECT_ADDITIONAL_PROPERTIES_ERROR",
    }

    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        raise ValueError(
            f"❌ Missing required environment variables: {', '.join(missing_vars)}. "
            f"❌ Please add these variables to your .env file."
        )


def load_environment() -> None:
    """Load environment variables and configure logging."""
    # Load environment variables from .env file
    load_dotenv()

    # Initialize configuration
    config = Config()
    config.update_from_env()

    # Update global variables
    globals().update(dict(config.__dict__.items()))

    # Configure logging
    logger.setLevel(getattr(logging, config.LOG_LEVEL))


# Initialize environment and validate required variables
validate_required_env_vars()
ERROR_MESSAGES = ErrorMessages()
load_environment()
