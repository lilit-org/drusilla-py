"""
This module contains API configurations, environment settings,
and default values.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

from dotenv import find_dotenv, load_dotenv

########################################################
#       Type Variables
########################################################

T = TypeVar("T")


########################################################
#       Base Config Class
########################################################


@dataclass
class BaseConfig:
    BOOLEAN_TRUE_VALUES: ClassVar[set[str]] = {"true", "1"}
    BOOLEAN_FALSE_VALUES: ClassVar[set[str]] = {"false", "0"}

    def _convert_env_value(self, value: str, field_type: type, current_value: Any) -> Any:
        if field_type is bool:
            value_lower = str(value).lower()
            return (
                True
                if value_lower in self.BOOLEAN_TRUE_VALUES
                else False if value_lower in self.BOOLEAN_FALSE_VALUES else current_value
            )

        try:
            if field_type in (tuple, set) and isinstance(current_value, field_type):
                if value.startswith(("(", "{")) and value.endswith((")", "}")):
                    try:
                        return eval(value)
                    except (SyntaxError, TypeError):
                        return current_value
                items = {item.strip() for item in value.split(",") if item.strip()}
                if field_type is tuple and len(items) == 2:
                    return tuple(items)
                if field_type is set:
                    return items
                return current_value
            return field_type(value)
        except (ValueError, TypeError):
            return current_value

    @staticmethod
    def get_env_var(name: str, default: T, type_func: type[T] = str) -> T:
        if (value := os.getenv(name)) is None:
            return default

        if type_func is bool:
            value_lower = str(value).lower()
            valid_values = BaseConfig.BOOLEAN_TRUE_VALUES | BaseConfig.BOOLEAN_FALSE_VALUES
            if value_lower in valid_values:
                return value_lower in BaseConfig.BOOLEAN_TRUE_VALUES
            return default

        try:
            return type_func(value)
        except (ValueError, TypeError):
            return default

    def update_from_env(self) -> None:
        for field_name, field_info in self.__dataclass_fields__.items():
            if field_name in os.environ and hasattr(self, field_name):
                value = self._convert_env_value(
                    os.environ[field_name], field_info.type, getattr(self, field_name)
                )
                setattr(self, field_name, value)

    @classmethod
    def validate_all(cls, *config_classes: type[BaseConfig]) -> None:
        missing_vars = []
        invalid_vars = []

        for config_class in config_classes:
            # Check required fields
            required = getattr(config_class, "REQUIRED_FIELDS", set())
            missing_vars.extend(f for f in required if f not in os.environ)

            # Validate default fields
            for field_name, default_value in getattr(config_class, "DEFAULTS", {}).items():
                if field_name in os.environ:
                    try:
                        value = os.environ[field_name]
                        field_type = type(default_value)

                        if field_type is bool:
                            valid_values = cls.BOOLEAN_TRUE_VALUES | cls.BOOLEAN_FALSE_VALUES
                            if value.lower() not in valid_values:
                                invalid_vars.append(f"{field_name} (invalid boolean value)")
                        elif field_type in (tuple, set):
                            if not value:
                                msg = f"{field_name} (empty {field_type.__name__} value)"
                                invalid_vars.append(msg)
                        elif field_name != "API_KEY" and not value:
                            invalid_vars.append(f"{field_name} (empty value)")
                        elif field_type in (int, float):
                            field_type(value)
                    except (ValueError, TypeError) as e:
                        invalid_vars.append(f"{field_name} ({str(e)})")

        errors = []
        if missing_vars:
            msg = f"❌ Missing required environment variables: {', '.join(missing_vars)}"
            errors.append(msg)
        if invalid_vars:
            msg = f"❌ Invalid environment variable values: {', '.join(invalid_vars)}"
            errors.append(msg)

        if errors:
            msg = "\n".join(errors) + "\n❌ Please check your .env file and fix these issues."
            raise ValueError(msg)


########################################################
#       Error Messages and Class
########################################################


ERROR_MESSAGES = {
    "SWORD_ERROR": "{error}",
    "RUNCONTEXT_ERROR": "{error}",
    "SHIELD_ERROR": "{error}",
    "RUNNER_ERROR": "{error}",
    "ORBS_ERROR": "{error}",
    "AGENT_EXEC_ERROR": "{error}",
    "MODEL_ERROR": "{error}",
    "TYPES_ERROR": "{error}",
    "NETWORK_ERROR": "{error}",
    "USAGE_ERROR": "{error}",
    "OBJECT_ADDITIONAL_PROPERTIES_ERROR": (
        "Object types cannot allow additional properties. This may be due to using an "
        "older Pydantic version or explicit configuration. If needed, update the "
        "function or output sword to use a non-strict schema."
    ),
}


@dataclass
class ErrMsg(BaseConfig):
    REQUIRED_FIELDS: ClassVar[set[str]] = set()

    def __post_init__(self) -> None:
        for key in ERROR_MESSAGES:
            setattr(self, key, ERROR_MESSAGES[key])
            if (value := os.getenv(key)) is not None:
                setattr(self, key, value)


########################################################
#       Config Class
########################################################


@dataclass
class Config(BaseConfig):
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "LOG_LEVEL": "DEBUG",
        "BASE_URL": "http://localhost:11434",
        "API_KEY": "",
        "MODEL": "deepseek-r1",
        "USER_AGENT": "Agents/Python",
        "MAX_TURNS": 10,
        "MAX_QUEUE_SIZE": 1000,
        "MAX_GUARDRAIL_QUEUE_SIZE": 100,
        "MAX_SHIELD_QUEUE_SIZE": 1000,
        "LRU_CACHE_SIZE": 128,
        "HTTP_TIMEOUT_TOTAL": 120.0,
        "HTTP_TIMEOUT_CONNECT": 30.0,
        "HTTP_TIMEOUT_READ": 90.0,
        "HTTP_MAX_KEEPALIVE_CONNECTIONS": 5,
        "HTTP_MAX_CONNECTIONS": 10,
        "CHAT_COMPLETIONS_ENDPOINT": "/api/chat",
        "THINK_TAGS": ("<think>", "</think>"),
        "SUPPORTED_LANGUAGES": {
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "ko",
            "zh",
            "ar",
            "hi",
            "nl",
            "pl",
            "tr",
            "vi",
            "he",
        },
    }
    REQUIRED_FIELDS: ClassVar[set[str]] = {"API_KEY"}

    def __post_init__(self) -> None:
        for field_name, default_value in self.DEFAULTS.items():
            setattr(self, field_name, default_value)
        self.update_from_env()


########################################################
#       Load Config and Error Messages
########################################################


def load_config() -> tuple[Config, ErrMsg]:
    {k: v for k, v in os.environ.items() if k not in Config.DEFAULTS and k not in ERROR_MESSAGES}

    if env_path := find_dotenv():
        load_dotenv(env_path, override=True)
        error_msgs_path = os.path.join(os.path.dirname(env_path), ".error_messages")
        if os.path.exists(error_msgs_path):
            load_dotenv(error_msgs_path, override=True)

    config, error_msgs = Config(), ErrMsg()
    BaseConfig.validate_all(Config, ErrMsg)
    return config, error_msgs


########################################################
#       Initialize Module
########################################################

config, err = load_config()

UNSET = object()
FAKE_RESPONSES_ID = "fake-responses-id"
HEADERS = {"User-Agent": config.USER_AGENT}
logger = logging.getLogger("drusilla.models")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(getattr(logging, config.LOG_LEVEL))


__all__ = ["config", "err", "HEADERS", "UNSET", "logger", "FAKE_RESPONSES_ID"]
