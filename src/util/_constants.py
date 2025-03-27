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
    "UNSET",
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
    "DEFAULT_WRAPPER_DICT_KEY",
    "DEFAULT_MAX_TURNS",
    "MAX_QUEUE_SIZE",
    "MAX_GUARDRAIL_QUEUE_SIZE",
    "CHAT_COMPLETIONS_ENDPOINT",
    "SUPPORTED_LANGUAGES",
    "HTTP_TIMEOUT_TOTAL",
    "HTTP_TIMEOUT_CONNECT",
    "HTTP_TIMEOUT_READ",
    "HTTP_MAX_KEEPALIVE_CONNECTIONS",
    "HTTP_MAX_CONNECTIONS",
    "LRU_CACHE_SIZE",
]

# Type Definitions
Environment = Literal["mac", "windows", "ubuntu", "browser"]
Button = Literal["left", "right", "wheel", "back", "forward"]
IncludeLiteral = Literal[
    "file_search_call.results",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
]

# Special Constants
UNSET = object()
FAKE_RESPONSES_ID = "__fake_id__"

# API Configuration
_USER_AGENT = f"Agents/Python {__version__}"
HEADERS = {"User-Agent": _USER_AGENT}
CHAT_COMPLETIONS_ENDPOINT = "/api/chat"

# Default Settings
DEFAULT_MODEL = "deepseek-r1:latest"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_WRAPPER_DICT_KEY = "response"
DEFAULT_MAX_TURNS = 10

# Queue and Cache Limits
MAX_QUEUE_SIZE = 1000
MAX_GUARDRAIL_QUEUE_SIZE = 100
LRU_CACHE_SIZE = 128

# HTTP Client Configuration
HTTP_TIMEOUT_TOTAL = 120.0
HTTP_TIMEOUT_CONNECT = 30.0
HTTP_TIMEOUT_READ = 90.0
HTTP_MAX_KEEPALIVE_CONNECTIONS = 5
HTTP_MAX_CONNECTIONS = 10

# Language Support
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "tr": "Turkish",
    "hr": "Croatian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "cs": "Czech",
    "sk": "Slovak",
    "bg": "Bulgarian",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
}
