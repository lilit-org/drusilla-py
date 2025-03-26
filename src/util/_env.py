import os
from typing import TypeVar, Dict, Any
from functools import lru_cache

########################################################
#              Constants
########################################################

T = TypeVar("T")
_env_cache: Dict[str, Any] = {}
TRUE_VALUES = {"true", "1", "yes"}

########################################################
#              Getters
########################################################

@lru_cache(maxsize=128)
def get_env_var(key: str, default: T) -> T:
    """Retrieves and converts an environment variable to the specified type.

    This function attempts to read an environment variable and converts its value to match
    the type of the default value. If the environment variable is not found or is empty,
    the default value is returned.

    Type conversion is handled automatically based on the default value's type:
    - For booleans: converts "true", "1", or "yes" (case-insensitive) to True
    - For integers: converts string to integer
    - For floats: converts string to float
    - For other types: returns the value as-is

    Args:
        key: The name of the environment variable to retrieve
        default: The default value to return if the environment variable is not found or empty

    Returns:
        The environment variable value converted to the appropriate type, or the default value
    """
    if key in _env_cache:
        return _env_cache[key]

    value = os.getenv(key)
    if value is None:
        _env_cache[key] = default
        return default

    try:
        if isinstance(default, bool):
            result = value.lower() in TRUE_VALUES
        elif isinstance(default, int):
            result = int(value)
        elif isinstance(default, float):
            result = float(value)
        else:
            result = value

        _env_cache[key] = result
        return result
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to convert environment variable '{key}' to type {type(default).__name__}: {e}")
