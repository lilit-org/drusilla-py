from typing_extensions import Literal

from ..models import shared
from ..util._types import AsyncDeepSeek

########################################################
#             Setters
########################################################

def set_default_model_key(key: str) -> None:
    shared.set_default_model_key(key)


def set_default_model_client(client: AsyncDeepSeek) -> None:
    shared.set_default_model_client(client)


def set_default_model_api(api: Literal["chat_completions", "responses"]) -> None:
    if api == "chat_completions":
        shared.set_use_responses_by_default(False)
    else:
        shared.set_use_responses_by_default(True)
