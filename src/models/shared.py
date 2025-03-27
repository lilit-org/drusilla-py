from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..util._types import AsyncDeepSeek


@dataclass
class SharedConfig:
    """Shared configuration for the DeepSeek framework."""

    _model_key: str | None = None
    _model_client: AsyncDeepSeek | None = None
    _use_responses: bool = False
    _instance: ClassVar[SharedConfig | None] = None

    @classmethod
    def get_instance(cls) -> SharedConfig:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def model_key(self) -> str | None:
        return self._model_key

    @model_key.setter
    def model_key(self, key: str) -> None:
        self._model_key = key

    @property
    def model_client(self) -> AsyncDeepSeek | None:
        return self._model_client

    @model_client.setter
    def model_client(self, client: AsyncDeepSeek) -> None:
        self._model_client = client

    @property
    def use_responses(self) -> bool:
        return self._use_responses

    @use_responses.setter
    def use_responses(self, use_responses: bool) -> None:
        self._use_responses = use_responses


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
