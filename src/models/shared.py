from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..util._types import AsyncDeepSeek

#######################################################
#              Data class for Shared Config
#######################################################


@dataclass
class SharedConfig:
    """Shared configuration for the DeepSeek framework."""

    model_key: str | None = None
    model_client: AsyncDeepSeek | None = None
    use_responses: bool = False
    _instance: ClassVar[SharedConfig | None] = None

    @classmethod
    def get_instance(cls) -> SharedConfig:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def set_default_model_key(key: str) -> None:
    SharedConfig.get_instance().model_key = key


def get_default_model_key() -> str | None:
    return SharedConfig.get_instance().model_key


def set_default_model_client(client: AsyncDeepSeek) -> None:
    SharedConfig.get_instance().model_client = client


def get_default_model_client() -> AsyncDeepSeek | None:
    return SharedConfig.get_instance().model_client


def set_use_responses_by_default(use_responses: bool) -> None:
    SharedConfig.get_instance().use_responses = use_responses


def get_use_responses_by_default() -> bool:
    return SharedConfig.get_instance().use_responses
