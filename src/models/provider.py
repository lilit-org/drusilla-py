from __future__ import annotations

import httpx

from src.util._constants import DEFAULT_BASE_URL, DEFAULT_MODEL
from src.util._env import get_env_var

from ..util._http import DefaultAsyncHttpxClient
from ..util._types import AsyncDeepSeek
from . import shared
from .chat_completions import ModelChatCompletionsModel
from .interface import Model
from .interface import ModelProvider as BaseModelProvider
from .responses import ModelResponsesModel

########################################################
#               Constants                                #
########################################################

MODEL = get_env_var("MODEL", DEFAULT_MODEL)
BASE_URL = get_env_var("BASE_URL", DEFAULT_BASE_URL)
_http_client: httpx.AsyncClient | None = None


########################################################
#               Private Methods                        #
########################################################
def shared_http_client() -> httpx.AsyncClient:
    global _http_client
    return _http_client or DefaultAsyncHttpxClient()


########################################################
#               Main Class                            #
########################################################


class ModelProvider(BaseModelProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_client: AsyncDeepSeek | None = None,
        organization: str | None = None,
        project: str | None = None,
        use_responses: bool | None = None,
    ) -> None:
        """Initialize a Model provider.

        Args:
            api_key: API key for the client. Uses default if not provided.
            base_url: Base URL for the client. Uses env var if not provided.
            model_client: Optional pre-configured client.
            organization: Organization ID for the client.
            project: Project ID for the client.
            use_responses: Whether to use responses API.
        """
        if model_client is not None:
            assert (
                api_key is None and base_url is None
            ), "Don't provide api_key or base_url if you provide model_client"
            self._client: AsyncDeepSeek | None = model_client
        else:
            self._client = None
            self._stored_api_key = api_key
            self._stored_base_url = base_url or BASE_URL
            self._stored_organization = organization
            self._stored_project = project

        self._use_responses = (
            use_responses
            if use_responses is not None
            else shared.get_use_responses_by_default()
        )

    def _get_client(self) -> AsyncDeepSeek:
        """Lazy load the client to avoid API key errors if never used."""
        if self._client is None:
            self._client = shared.get_default_model_client() or AsyncDeepSeek(
                api_key=self._stored_api_key or shared.get_default_model_key(),
                base_url=self._stored_base_url,
                organization=self._stored_organization,
                project=self._stored_project,
                http_client=shared_http_client(),
            )
        return self._client

    def get_model(self, model_name: str | None) -> Model:
        """Get a model instance based on name and response type."""
        model_name = model_name or MODEL

        client = self._get_client()
        return (
            ModelResponsesModel(model=model_name, model_client=client)
            if self._use_responses
            else ModelChatCompletionsModel(model=model_name, model_client=client)
        )
