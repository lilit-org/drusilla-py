"""
Constants used throughout the deepseek agentic framework.
This module contains API configurations, environment settings, and default values.
"""

from __future__ import annotations

import os
import logging
import sys
from typing import Set
from dotenv import load_dotenv

__all__ = (
    "HEADERS",
    "UNSET",
    "FAKE_RESPONSES_ID",
    "LOG_LEVEL",
    "BASE_URL",
    "API_KEY",
    "MODEL",
    "MAX_TURNS",
    "MAX_QUEUE_SIZE",
    "MAX_GUARDRAIL_QUEUE_SIZE",
    "MAX_SHIELD_QUEUE_SIZE",
    "LRU_CACHE_SIZE",
    "CHAT_COMPLETIONS_ENDPOINT",
    "HTTP_TIMEOUT_TOTAL",
    "HTTP_TIMEOUT_CONNECT",
    "HTTP_TIMEOUT_READ",
    "HTTP_MAX_KEEPALIVE_CONNECTIONS",
    "HTTP_MAX_CONNECTIONS",
    "SUPPORTED_LANGUAGES",
)

# Constants
UNSET = object()
FAKE_RESPONSES_ID = "fake_responses"

# Supported languages for translation
SUPPORTED_LANGUAGES: Set[str] = {
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "pt",  # Portuguese
    "ru",  # Russian
    "ja",  # Japanese
    "ko",  # Korean
    "zh",  # Chinese
    "ar",  # Arabic
    "hi",  # Hindi
    "nl",  # Dutch
    "pl",  # Polish
    "tr",  # Turkish
    "vi",  # Vietnamese
    "he",  # Hebrew
}

# Load environment variables from .env file
load_dotenv()

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logger = logging.getLogger("deepseek.agents")
logger.setLevel(getattr(logging, LOG_LEVEL))
logger.addHandler(logging.StreamHandler(sys.stdout))

# Connection
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434")
API_KEY = os.getenv("API_KEY", "NONE")
MODEL = os.getenv("MODEL", "deepseek-r1")

# Model logic and Optimizations
MAX_TURNS = int(os.getenv("MAX_TURNS", str(10)))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "1000"))
MAX_GUARDRAIL_QUEUE_SIZE = int(os.getenv("MAX_GUARDRAIL_QUEUE_SIZE", "100"))
MAX_SHIELD_QUEUE_SIZE = 1000
LRU_CACHE_SIZE = int(os.getenv("LRU_CACHE_SIZE", "128"))

# HTTP Client Configuration
HTTP_TIMEOUT_TOTAL = float(os.getenv("HTTP_TIMEOUT_TOTAL", "120.0"))
HTTP_TIMEOUT_CONNECT = float(os.getenv("HTTP_TIMEOUT_CONNECT", "30.0"))
HTTP_TIMEOUT_READ = float(os.getenv("HTTP_TIMEOUT_READ", "90.0"))
HTTP_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "5"))
HTTP_MAX_CONNECTIONS = int(os.getenv("HTTP_MAX_CONNECTIONS", "10"))

# API Configuration
_USER_AGENT = f"Agents/Python"
HEADERS = {"User-Agent": _USER_AGENT}
CHAT_COMPLETIONS_ENDPOINT = "/api/chat"

