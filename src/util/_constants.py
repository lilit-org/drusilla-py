from __future__ import annotations

from typing import Literal

from ._version import __version__

########################################################
#              Constants                               #
########################################################

# API and Response Constants
FAKE_RESPONSES_ID = "__fake_id__"
_USER_AGENT = f"Agents/Python {__version__}"
_HEADERS = {"User-Agent": _USER_AGENT}

# Environment Constants
Environment = Literal["mac", "windows", "ubuntu", "browser"]
Button = Literal["left", "right", "wheel", "back", "forward"]

# API Response Include Literals
IncludeLiteral = Literal[
    "file_search_call.results",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
]

# Sentinel Values
NOT_GIVEN = object()

# Default Values
DEFAULT_MODEL = "deepseek-r1"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_WRAPPER_DICT_KEY = "response"
DEFAULT_MAX_TURNS = 10 