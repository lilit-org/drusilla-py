"""
Model Provider Management

This module provides the ModelProvider class, a flexible provider interface for managing
model instances and their configurations. It handles client initialization, API key
management, and model selection with support for both chat completions and responses
API modes.

Key features:
- Dynamic client initialization with environment-based configuration
- API key and base URL management with fallback to environment variables
- Support for both chat completions and responses API modes
- Organization and project context management
- Lazy client loading to prevent premature API key validation
"""

from __future__ import annotations

from src.network.client import DeepSeekClient
from src.network.http import DefaultAsyncHttpxClient
from src.util.constants import BASE_URL, MODEL
from src.util.types import AsyncDeepSeek

from .chat import ModelChatCompletionsModel
from .interface import Model
from .interface import ModelProvider as BaseModelProvider
from .responses import ModelResponsesModel

########################################################
#               Main Class: Model Provider
########################################################


class ModelProvider(BaseModelProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        use_responses: bool = False,
    ) -> None:
        self._stored_api_key = api_key
        self._stored_base_url = base_url or BASE_URL
        self._stored_organization = organization
        self._stored_project = project
        self._use_responses = use_responses
        self._client: AsyncDeepSeek | None = None

    @property
    def use_responses(self) -> bool:
        return self._use_responses

    @use_responses.setter
    def use_responses(self, value: bool) -> None:
        self._use_responses = value

    def _get_client(self) -> AsyncDeepSeek:
        """Lazy load the client to avoid API key errors if never used."""
        if self._client is None:
            self._client = DeepSeekClient(
                api_key=self._stored_api_key,
                base_url=self._stored_base_url,
                organization=self._stored_organization,
                project=self._stored_project,
                http_client=DefaultAsyncHttpxClient(),
            )
        return self._client

    def get_model(self, model_name: str | None = None) -> Model:
        """Get a model instance based on name and response type."""
        model_name = model_name or MODEL
        client = self._get_client()

        if self._use_responses:
            return ModelResponsesModel(model=model_name, model_client=client)
        return ModelChatCompletionsModel(model=model_name, model_client=client)
