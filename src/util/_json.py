import re
from typing import TypeVar

from pydantic import TypeAdapter, ValidationError

from ._exceptions import ModelError

########################################################
#              Constants
########################################################

T = TypeVar("T")


########################################################
#              Validators
########################################################

def validate_json(json_str: str, type_adapter: TypeAdapter[T], partial: bool = False) -> T:
    """Validates a JSON string against a type adapter. Raises ModelError if invalid."""
    try:
        return type_adapter.validate_json(json_str, strict=not partial)
    except ValidationError as e:
        raise ModelError(f"Invalid JSON: {e}") from e


########################################################
#              String Transformers
########################################################

def transform_string_function_style(name: str) -> str:
    """Converts a string into a valid Python function name.

    Args:
        name: Input string to transform

    Returns:
        Lowercase string with spaces and special chars replaced by underscores
    """
    return re.sub(r"[^a-zA-Z0-9]", "_", name.replace(" ", "_")).lower()
