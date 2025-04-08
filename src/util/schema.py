"""
This module provides utilities for enforcing strict JSON schema validation
rules and managing schema references.

Key features:
- Enforces strict object property validation by default
- Recursively processes and validates nested schema definitions
- Handles schema references ($ref) resolution with caching
- Supports logical operators (anyOf, allOf)
- Ensures consistent schema structure and validation rules
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Final, TypeAlias, cast, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

from .constants import LRU_CACHE_SIZE, UNSET
from .exceptions import ModelError, UsageError, format_error_message

########################################################
#             Type Aliases and Constants                #
########################################################
JSONSchema: TypeAlias = dict[str, Any]
SchemaPath: TypeAlias = tuple[str, ...]

EMPTY_JSON_SCHEMA: Final[JSONSchema] = {
    "additionalProperties": False,
    "type": "object",
    "properties": {},
    "required": [],
}

CACHE_SIZE: Final[int] = LRU_CACHE_SIZE
LOGICAL_OPERATORS: Final[tuple[str, ...]] = ("anyOf", "allOf")
SCHEMA_DEFINITION_KEYS: Final[tuple[str, ...]] = ("definitions", "$defs")

# Error message templates
TYPES_ERROR_MESSAGE: Final[str] = "Type error: {error}"
OBJECT_ADDITIONAL_PROPERTIES_ERROR: Final[str] = (
    "Object types cannot allow additional properties. This may be due to using an "
    "older Pydantic version or explicit configuration. If needed, update the function "
    "or output sword to use a non-strict schema."
)

########################################################
#               Private Methods
########################################################


def _make_hashable(d: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Convert a dictionary to a hashable tuple representation."""
    return tuple((k, _make_hashable(v) if isinstance(v, dict) else v) for k, v in sorted(d.items()))


@lru_cache(maxsize=CACHE_SIZE)
def _resolve_schema_ref_cached(*, root_hash: tuple[tuple[str, Any], ...], ref: str) -> JSONSchema:
    """Resolve a JSON schema reference to its target schema with caching."""
    if not ref.startswith("#/"):
        raise ModelError(
            format_error_message(
                ValueError(f"Invalid $ref format {ref!r}; must start with #/"), TYPES_ERROR_MESSAGE
            )
        )

    try:
        resolved = dict(_make_dict(root_hash))
        for key in ref[2:].split("/"):
            resolved = resolved[key]
            if not isinstance(resolved, dict):
                raise ModelError(
                    format_error_message(
                        ValueError(
                            f"Invalid resolution path for {ref} - "
                            f"encountered non-dictionary at {resolved}"
                        ),
                        TYPES_ERROR_MESSAGE,
                    )
                )
        return cast(JSONSchema, resolved)
    except KeyError as e:
        raise ModelError(format_error_message(e, TYPES_ERROR_MESSAGE, {"ref": ref})) from e


def _make_dict(t: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Convert a hashable tuple representation back to a dictionary."""
    return {k: _make_dict(v) if isinstance(v, tuple) else v for k, v in t}


def _enforce_strict_schema_rules(
    schema: JSONSchema,
    *,
    path: SchemaPath,
    root: JSONSchema,
) -> JSONSchema:
    """Enforces strict JSON schema rules by recursively validating and modifying the schema."""
    # Process nested definitions
    for def_key in SCHEMA_DEFINITION_KEYS:
        if def_key in schema and isinstance(schema[def_key], dict):
            schema[def_key] = {
                name: _enforce_strict_schema_rules(
                    def_schema, path=(*path, def_key, name), root=root
                )
                for name, def_schema in schema[def_key].items()
            }

    # Handle object type properties
    if schema.get("type") == "object":
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        elif schema["additionalProperties"] is True:
            raise UsageError(OBJECT_ADDITIONAL_PROPERTIES_ERROR)

    # Process object properties
    if "properties" in schema and isinstance(schema["properties"], dict):
        properties = schema["properties"]
        schema["required"] = list(properties.keys())
        schema["properties"] = {
            key: _enforce_strict_schema_rules(
                prop_schema, path=(*path, "properties", key), root=root
            )
            for key, prop_schema in properties.items()
        }

    # Process array items
    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _enforce_strict_schema_rules(
            schema["items"], path=(*path, "items"), root=root
        )

    # Process logical operators
    for operator in LOGICAL_OPERATORS:
        if operator in schema and isinstance(schema[operator], list):
            values = schema[operator]
            if operator == "allOf" and len(values) == 1:
                schema.update(
                    _enforce_strict_schema_rules(values[0], path=(*path, operator, "0"), root=root)
                )
                del schema[operator]
            else:
                schema[operator] = [
                    _enforce_strict_schema_rules(entry, path=(*path, operator, str(i)), root=root)
                    for i, entry in enumerate(values)
                ]

    # Clean up default values
    if schema.get("default", UNSET) is None:
        del schema["default"]

    # Handle schema references
    if ref := schema.get("$ref"):
        if not isinstance(ref, str):
            raise ModelError(
                format_error_message(
                    TypeError(f"$ref must be a string, got {ref}"), TYPES_ERROR_MESSAGE
                )
            )

        if len(schema) > 1:
            root_hash = _make_hashable(root)
            resolved = _resolve_schema_ref_cached(root_hash=root_hash, ref=ref)
            schema.update({**resolved, **schema})
            del schema["$ref"]
            schema = _enforce_strict_schema_rules(schema, path=path, root=root)

    return schema


@lru_cache(maxsize=CACHE_SIZE)
def is_subclass_of_base_model_or_dict(t: Any) -> bool:
    """Check if a type is a subclass of BaseModel or dict."""
    if not isinstance(t, type):
        return False
    origin = get_origin(t)
    return issubclass(origin or t, BaseModel | dict)


@lru_cache(maxsize=CACHE_SIZE)
def type_to_str(t: type[Any]) -> str:
    """Convert a type to its string representation."""
    origin = get_origin(t)
    args = get_args(t)

    if origin is None:
        return t.__name__
    args_str = ", ".join(type_to_str(arg) for arg in args)
    return f"{origin.__name__}[{args_str}]" if args else str(t)


@lru_cache(maxsize=CACHE_SIZE)
def get_type_adapter(output_type: type[Any]) -> TypeAdapter[Any]:
    """Get or create a type adapter with caching."""
    return TypeAdapter(output_type)


########################################################
#               Public Methods
########################################################


def ensure_strict_json_schema(schema: JSONSchema) -> JSONSchema:
    """Ensure a JSON schema follows strict rules."""
    if not schema:
        return EMPTY_JSON_SCHEMA
    return _enforce_strict_schema_rules(schema, path=(), root=schema)


def resolve_schema_ref(*, root: JSONSchema, ref: str) -> JSONSchema:
    """Resolves a JSON schema reference to its target schema."""
    root_hash = _make_hashable(root)
    return _resolve_schema_ref_cached(root_hash=root_hash, ref=ref)
