from __future__ import annotations

from ..util._types import AsyncDeepSeek

########################################################
#            Shared variables                          #
########################################################
_default_model_key: str | None = None
_default_model_client: AsyncDeepSeek | None = None
_use_responses_by_default: bool = False


########################################################
#            Setters and getters                      #
########################################################

def set_default_model_key(key: str) -> None:
    global _default_model_key
    _default_model_key = key


def get_default_model_key() -> str | None:
    return _default_model_key


def set_default_model_client(client: AsyncDeepSeek) -> None:
    global _default_model_client
    _default_model_client = client


def get_default_model_client() -> AsyncDeepSeek | None:
    return _default_model_client


def set_use_responses_by_default(use_responses: bool) -> None:
    global _use_responses_by_default
    _use_responses_by_default = use_responses


def get_use_responses_by_default() -> bool:
    return _use_responses_by_default
