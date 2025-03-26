from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypedDict, get_args, get_origin

from ..util import _json
from ..util._env import get_env_var
from ..util._exceptions import ModelError, UsageError
from ..util._strict_schema import ensure_strict_json_schema

########################################################
#             Constants                                #
########################################################

WRAPPER_DICT_KEY = get_env_var("WRAPPER_DICT_KEY", "response")


########################################################
#               Private Methods                        #
########################################################

def _is_subclass_of_base_model_or_dict(t: Any) -> bool:
    if not isinstance(t, type):
        return False
    origin = get_origin(t)
    allowed_types = (BaseModel, dict)
    return issubclass(origin or t, allowed_types)


def _type_to_str(t: type[Any]) -> str:
    origin = get_origin(t)
    args = get_args(t)

    if origin is None:
        return t.__name__
    elif args:
        args_str = ", ".join(_type_to_str(arg) for arg in args)
        return f"{origin.__name__}[{args_str}]"
    else:
        return str(t)


########################################################
#               Main Class                            #
########################################################

@dataclass(init=False)
class AgentOutputSchema:
    """Schema for validating and parsing LLM output into specified types."""

    output_type: type[Any]
    _type_adapter: TypeAdapter[Any]
    _is_wrapped: bool
    _output_schema: dict[str, Any]
    strict_json_schema: bool

    def __init__(self, output_type: type[Any], strict_json_schema: bool = True):
        """
        Args:
            output_type: Type to validate against
            strict_json_schema: Enable strict validation (recommended)
        """
        self.output_type = output_type
        self.strict_json_schema = strict_json_schema

        if output_type is None or output_type is str:
            self._is_wrapped = False
            self._type_adapter = TypeAdapter(output_type)
            self._output_schema = self._type_adapter.json_schema()
            return

        self._is_wrapped = not _is_subclass_of_base_model_or_dict(output_type)

        if self._is_wrapped:
            OutputType = TypedDict(
                "OutputType",
                {
                    WRAPPER_DICT_KEY: output_type,
                },
            )
            self._type_adapter = TypeAdapter(OutputType)
            self._output_schema = self._type_adapter.json_schema()
        else:
            self._type_adapter = TypeAdapter(output_type)
            self._output_schema = self._type_adapter.json_schema()

        if self.strict_json_schema:
            self._output_schema = ensure_strict_json_schema(self._output_schema)

    def is_plain_text(self) -> bool:
        return self.output_type is None or self.output_type is str

    def json_schema(self) -> dict[str, Any]:
        if self.is_plain_text():
            raise UsageError("No JSON schema for plain text output")
        return self._output_schema

    def validate_json(self, json_str: str, partial: bool = False) -> Any:
        validated = _json.validate_json(json_str, self._type_adapter, partial)
        if self._is_wrapped:
            if not isinstance(validated, dict):
                raise ModelError(
                    f"Expected a dict, got {type(validated)} for JSON: {json_str}"
                )

            if WRAPPER_DICT_KEY not in validated:
                raise ModelError(
                    f"Could not find key {WRAPPER_DICT_KEY} in JSON: {json_str}"
                )
            return validated[WRAPPER_DICT_KEY]
        return validated

    def output_type_name(self) -> str:
        return _type_to_str(self.output_type)
