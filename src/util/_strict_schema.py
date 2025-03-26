from __future__ import annotations

from typing import Any

from typing_extensions import TypeGuard

from ._constants import NOT_GIVEN
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

########################################################
#               Private Functions                      #
########################################################

def _enforce_strict_schema_rules(
    schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    """Enforces strict JSON schema rules by recursively validating and modifying the schema."""
    if not is_dict(schema):
        raise TypeError(f"Schema must be a dictionary, got {schema}; path={path}")

    # Process nested definitions
    for def_key in ["$defs", "definitions"]:
        definitions = schema.get(def_key)
        if is_dict(definitions):
            for name, def_schema in definitions.items():
                _enforce_strict_schema_rules(def_schema, path=(*path, def_key, name), root=root)

    # Handle object type properties
    schema_type = schema.get("type")
    if schema_type == "object":
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        elif schema["additionalProperties"] is True:
            raise UsageError(
                "Object types cannot allow additional properties. This may be due to using an "
                "older Pydantic version or explicit configuration. If needed, update the function "
                "or output tool to use a non-strict schema."
            )

    # Process object properties
    properties = schema.get("properties")
    if is_dict(properties):
        schema["required"] = list(properties.keys())
        schema["properties"] = {
            key: _enforce_strict_schema_rules(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    # Process array items
    items = schema.get("items")
    if is_dict(items):
        schema["items"] = _enforce_strict_schema_rules(items, path=(*path, "items"), root=root)

    # Process logical operators
    for operator in ["anyOf", "allOf"]:
        values = schema.get(operator)
        if is_list(values):
            if operator == "allOf" and len(values) == 1:
                schema.update(_enforce_strict_schema_rules(values[0], path=(*path, operator, "0"), root=root))
                schema.pop(operator)
            else:
                schema[operator] = [
                    _enforce_strict_schema_rules(entry, path=(*path, operator, str(i)), root=root)
                    for i, entry in enumerate(values)
                ]

    # Remove null defaults
    if schema.get("default", NOT_GIVEN) is None:
        schema.pop("default")

    # Handle schema references
    ref = schema.get("$ref")
    if ref and has_more_than_n_keys(schema, 1):
        if not isinstance(ref, str):
            raise ValueError(f"$ref must be a string, got {ref}")

        resolved = resolve_schema_ref(root=root, ref=ref)
        if not is_dict(resolved):
            raise ValueError(f"$ref {ref} must resolve to a dictionary, got {resolved}")

        schema.update({**resolved, **schema})
        schema.pop("$ref")
        return _enforce_strict_schema_rules(schema, path=path, root=root)

    return schema


########################################################
#               Public Functions                       #
########################################################

def ensure_strict_json_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Enforce strict JSON schema standard. """
    if schema == {}:
        return EMPTY_JSON_SCHEMA
    return _enforce_strict_schema_rules(schema, path=(), root=schema)


def resolve_schema_ref(*, root: dict[str, object], ref: str) -> object:
    """Resolves a JSON schema reference to its target schema."""
    if not ref.startswith("#/"):
        raise ValueError(f"Invalid $ref format {ref!r}; must start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        if not is_dict(value):
            raise ValueError(f"Invalid resolution path for {ref} - encountered non-dictionary at {resolved}")
        resolved = value

    return resolved


def is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    """Type guard for dictionary objects."""
    return isinstance(obj, dict)


def is_list(obj: object) -> TypeGuard[list[object]]:
    """Type guard for list objects."""
    return isinstance(obj, list)


def has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    """Checks if a dictionary has more than n keys."""
    return len(obj) > n
