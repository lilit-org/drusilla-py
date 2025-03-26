import os
from typing import TypeVar

########################################################
#              Constants
########################################################

T = TypeVar("T")


########################################################
#              Getters
########################################################

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
    value = os.getenv(key) or default

    if isinstance(default, bool):
        return bool(value.lower() in ("true", "1", "yes"))
    elif isinstance(default, int):
        return int(value)
    elif isinstance(default, float):
        return float(value)
    else:
        return value
