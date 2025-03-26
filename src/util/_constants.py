from __future__ import annotations

from typing import Literal

from ._version import __version__

"""
Constants used throughout the deepseek agentic framework.
This module contains API configurations, environment settings, and default values.
"""

__all__ = [
    "FAKE_RESPONSES_ID",
    "HEADERS",
    "Environment",
    "Button",
    "IncludeLiteral",
    "NOT_GIVEN",
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
    "DEFAULT_WRAPPER_DICT_KEY",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MAX_QUEUE_SIZE",
    "DEFAULT_MAX_GUARDRAIL_QUEUE_SIZE",
]

# API Configuration
FAKE_RESPONSES_ID = "__fake_id__"
_USER_AGENT = f"Agents/Python {__version__}"
HEADERS = {"User-Agent": _USER_AGENT}

# Type Definitions
Environment = Literal["mac", "windows", "ubuntu", "browser"]
Button = Literal["left", "right", "wheel", "back", "forward"]
IncludeLiteral = Literal[
    "file_search_call.results",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
]

# System Constants
NOT_GIVEN = object()

# Default Configuration Values
DEFAULT_MODEL = "deepseek-r1"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_WRAPPER_DICT_KEY = "response"
DEFAULT_MAX_TURNS = 10
DEFAULT_MAX_QUEUE_SIZE = 1000
DEFAULT_MAX_GUARDRAIL_QUEUE_SIZE = 100
