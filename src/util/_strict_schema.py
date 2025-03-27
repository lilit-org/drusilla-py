from __future__ import annotations

from functools import lru_cache
from typing import Any, TypeAlias, cast

from typing_extensions import TypeGuard

from ._constants import LRU_CACHE_SIZE, UNSET
from ._exceptions import UsageError

########################################################
#               Private Constants                      #
########################################################

EMPTY_JSON_SCHEMA = {
    "additionalProperties": False,
    "type": "object",
    "properties": {},
    "required": [],
}

JSONSchema: TypeAlias = dict[str, Any]
SchemaPath: TypeAlias = tuple[str, ...]

########################################################
#               Private Functions                      #
########################################################

@lru_cache(maxsize=LRU_CACHE_SIZE)
def _resolve_schema_ref_cached(*, root: JSONSchema, ref: str) -> JSONSchema:
    """Cached version of schema reference resolution."""
    if not ref.startswith("#/"):
        raise ValueError(f"Invalid $ref format {ref!r}; must start with #/")

    resolved = root
    for key in (path := ref[2:].split("/")):
        resolved = resolved[key]
        if not is_dict(resolved):
            raise ValueError(f"Invalid resolution path for {ref} - encountered non-dictionary at {resolved}")

    return cast(JSONSchema, resolved)

def _enforce_strict_schema_rules(
    schema: JSONSchema,
    *,
    path: SchemaPath,
    root: JSONSchema,
) -> JSONSchema:
    """Enforces strict JSON schema rules by recursively validating and modifying the schema."""
    if not is_dict(schema):
        raise TypeError(f"Schema must be a dictionary, got {schema}; path={path}")

    # Process nested definitions
    for def_key in ["$defs", "definitions"]:
        if def_key in schema and is_dict(schema[def_key]):
            schema[def_key] = {
                name: _enforce_strict_schema_rules(def_schema, path=(*path, def_key, name), root=root)
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
                "or output tool to use a non-strict schema."
            )

    # Process object properties
    if "properties" in schema and is_dict(schema["properties"]):
        properties = schema["properties"]
        schema["required"] = list(properties.keys())
        schema["properties"] = {
            key: _enforce_strict_schema_rules(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    # Process array items
    if "items" in schema and is_dict(schema["items"]):
        schema["items"] = _enforce_strict_schema_rules(schema["items"], path=(*path, "items"), root=root)

    # Process logical operators
    for operator in ["anyOf", "allOf"]:
        if operator in schema and is_list(schema[operator]):
            values = schema[operator]
            if operator == "allOf" and len(values) == 1:
                schema.update(_enforce_strict_schema_rules(values[0], path=(*path, operator, "0"), root=root))
                schema.pop(operator)
            else:
                schema[operator] = [
                    _enforce_strict_schema_rules(entry, path=(*path, operator, str(i)), root=root)
                    for i, entry in enumerate(values)
                ]

    # Remove null defaults
    if schema.get("default", UNSET) is None:
        schema.pop("default")

    # Handle schema references
    if ref := schema.get("$ref"):
        if has_more_than_n_keys(schema, 1):
            if not isinstance(ref, str):
                raise ValueError(f"$ref must be a string, got {ref}")

            resolved = _resolve_schema_ref_cached(root=root, ref=ref)
            schema.update({**resolved, **schema})
            schema.pop("$ref")
            return _enforce_strict_schema_rules(schema, path=path, root=root)

    return schema

########################################################
#               Public Functions                       #
########################################################

def ensure_strict_json_schema(
    schema: JSONSchema,
) -> JSONSchema:
    """Enforce strict JSON schema standard."""
    if not schema:
        return EMPTY_JSON_SCHEMA
    return _enforce_strict_schema_rules(schema, path=(), root=schema)

def resolve_schema_ref(*, root: JSONSchema, ref: str) -> JSONSchema:
    """Resolves a JSON schema reference to its target schema."""
    return _resolve_schema_ref_cached(root=root, ref=ref)

def is_dict(obj: object) -> TypeGuard[JSONSchema]:
    """Type guard for dictionary objects."""
    return isinstance(obj, dict)

def is_list(obj: object) -> TypeGuard[list[object]]:
    """Type guard for list objects."""
    return isinstance(obj, list)

def has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    """Checks if a dictionary has more than n keys."""
    return len(obj) > n
