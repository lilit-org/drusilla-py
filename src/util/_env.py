import os
from functools import lru_cache
from typing import TypeVar

from ._constants import LRU_CACHE_SIZE
from ._exceptions import UsageError

########################################################
#              Constants
########################################################

T = TypeVar("T")
TRUE_VALUES = {"true", "1", "yes"}


########################################################
#              Getters
########################################################


@lru_cache(maxsize=LRU_CACHE_SIZE)
def get_env_var(key: str, default: T) -> T:
    """Retrieves and converts an environment variable to the specified type."""
    if value := os.getenv(key):
        try:
            if isinstance(default, bool):
                return value.lower() in {"true", "1", "yes"}
            if isinstance(default, int):
                return int(value)
            if isinstance(default, float):
                return float(value)
            return value
        except (ValueError, TypeError) as e:
            raise UsageError(
                f"Failed to convert env variable '{key}' to {type(default).__name__}: {e}"
            ) from e
    return default
