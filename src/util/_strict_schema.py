"""
This module provides utilities for enforcing strict JSON schema validation
rules and managing schema references.

Key features:
- Enforces strict object property validation by default
- Recursively processes and validates nested schema definitions
- Handles schema references ($ref) resolution with caching
- Supports logical operators (anyOf, allOf)
- Ensures consistent schema structure and validation rules

The module is particularly useful for ensuring type safety and strict validation
in JSON schema-based systems, preventing unexpected property additions and
maintaining consistent data structures.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Final, TypeAlias, cast

from ._constants import LRU_CACHE_SIZE, UNSET
from ._exceptions import ModelError, UsageError

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


########################################################
#               Private Methods
########################################################


@lru_cache(maxsize=CACHE_SIZE)
def _resolve_schema_ref_cached(*, root: JSONSchema, ref: str) -> JSONSchema:
    """Resolve a JSON schema reference to its target schema with caching."""
    if not ref.startswith("#/"):
        raise ModelError(f"Invalid $ref format {ref!r}; must start with #/")

    try:
        resolved = root
        for key in ref[2:].split("/"):
            resolved = resolved[key]
            if not isinstance(resolved, dict):
                raise ModelError(
                    f"Invalid resolution path for {ref} - encountered non-dictionary at {resolved}"
                )
        return cast(JSONSchema, resolved)
    except KeyError as e:
        raise ModelError(f"Invalid $ref path {ref}: {str(e)}") from e


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
            raise UsageError(
                "Object types cannot allow additional properties. This may be due to using an "
                "older Pydantic version or explicit configuration. If needed, update the function "
                "or output sword to use a non-strict schema."
            )

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
                schema.pop(operator)
            else:
                schema[operator] = [
                    _enforce_strict_schema_rules(entry, path=(*path, operator, str(i)), root=root)
                    for i, entry in enumerate(values)
                ]

    # Clean up default values
    if schema.get("default", UNSET) is None:
        schema.pop("default")

    # Handle schema references
    if ref := schema.get("$ref"):
        if not isinstance(ref, str):
            raise ModelError(f"$ref must be a string, got {ref}")

        if len(schema) > 1:
            resolved = _resolve_schema_ref_cached(root=root, ref=ref)
            schema.update({**resolved, **schema})
            schema.pop("$ref")
            schema = _enforce_strict_schema_rules(schema, path=path, root=root)

    return schema


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
    return _resolve_schema_ref_cached(root=root, ref=ref)
