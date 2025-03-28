from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypedDict, get_args, get_origin

from ..util import _json
from ..util._constants import DEFAULT_WRAPPER_DICT_KEY
from ..util._env import get_env_var
from ..util._exceptions import ModelError, UsageError
from ..util._strict_schema import ensure_strict_json_schema

########################################################
#             Constants                                #
########################################################

WRAPPER_DICT_KEY = get_env_var("WRAPPER_DICT_KEY", DEFAULT_WRAPPER_DICT_KEY)
LRU_CACHE_SIZE = int(get_env_var("LRU_CACHE_SIZE", "128"))


########################################################
#               Private Methods                        #
########################################################


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _is_subclass_of_base_model_or_dict(t: Any) -> bool:
    if not isinstance(t, type):
        return False
    origin = get_origin(t)
    return issubclass(origin or t, (BaseModel, dict))


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _type_to_str(t: type[Any]) -> str:
    origin = get_origin(t)
    args = get_args(t)

    if origin is None:
        return t.__name__
    args_str = ", ".join(_type_to_str(arg) for arg in args)
    return f"{origin.__name__}[{args_str}]" if args else str(t)


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _get_type_adapter(
    output_type: type[Any], is_wrapped: bool = False
) -> TypeAdapter[Any]:
    """Get or create a type adapter with caching."""
    if output_type is None or output_type is str:
        return TypeAdapter(output_type)

    if is_wrapped:
        OutputType = TypedDict(
            "OutputType",
            {
                WRAPPER_DICT_KEY: output_type,
            },
        )
        return TypeAdapter(OutputType)

    return TypeAdapter(output_type)


########################################################
#             Main Class for Output Schema            #
########################################################


@dataclass(init=False)
class AgentOutputSchema:
    """Schema for validating and parsing LLM output into specified types."""

    output_type: type[Any]
    _type_adapter: TypeAdapter[Any]
    _is_wrapped: bool
    _output_schema: dict[str, Any]
    strict_json_schema: bool
    _is_plain_text: ClassVar[bool] = False
    _type_name: ClassVar[str] = ""

    def __init__(self, output_type: type[Any], strict_json_schema: bool = True):
        """
        Args:
            output_type: Type to validate against
            strict_json_schema: Enable strict validation (recommended)
        """
        self.output_type = output_type
        self.strict_json_schema = strict_json_schema
        self._is_plain_text = output_type is None or output_type is str
        self._type_name = _type_to_str(output_type)

        if self._is_plain_text:
            self._is_wrapped = False
            self._type_adapter = TypeAdapter(output_type)
            self._output_schema = self._type_adapter.json_schema()
            return

        self._is_wrapped = not _is_subclass_of_base_model_or_dict(output_type)
        self._type_adapter = _get_type_adapter(output_type, self._is_wrapped)
        self._output_schema = self._type_adapter.json_schema()

        if self.strict_json_schema:
            self._output_schema = ensure_strict_json_schema(self._output_schema)

    def is_plain_text(self) -> bool:
        return self._is_plain_text

    def json_schema(self) -> dict[str, Any]:
        if self.is_plain_text():
            raise UsageError("No JSON schema for plain text output")
        return self._output_schema

    def validate_json(self, json_str: str, partial: bool = False) -> Any:
        validated = _json.validate_json(json_str, self._type_adapter, partial)
        if (
            self._is_wrapped
            and isinstance(validated, dict)
            and WRAPPER_DICT_KEY not in validated
        ):
            raise ModelError(
                f"Could not find key {WRAPPER_DICT_KEY} in JSON: {json_str}"
            )
        return (
            validated[WRAPPER_DICT_KEY]
            if self._is_wrapped and isinstance(validated, dict)
            else validated
        )

    def output_type_name(self) -> str:
        return self._type_name
