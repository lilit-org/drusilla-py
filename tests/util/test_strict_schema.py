import pytest

from src.util._strict_schema import (
    EMPTY_JSON_SCHEMA,
    ModelError,
    UsageError,
    ensure_strict_json_schema,
    resolve_schema_ref,
)


def test_empty_schema():
    """Test handling of empty schema."""
    result = ensure_strict_json_schema({})
    assert result == EMPTY_JSON_SCHEMA


def test_basic_object_schema():
    """Test basic object schema enforcement."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }

    result = ensure_strict_json_schema(schema)

    assert result["type"] == "object"
    assert "additionalProperties" in result
    assert result["additionalProperties"] is False
    assert "name" in result["properties"]
    assert "age" in result["properties"]
    assert "name" in result["required"]
    assert "age" in result["required"]


def test_nested_object_schema():
    """Test nested object schema enforcement."""
    schema = {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            }
        },
    }

    result = ensure_strict_json_schema(schema)

    assert result["type"] == "object"
    assert result["additionalProperties"] is False
    assert "person" in result["properties"]
    assert "person" in result["required"]

    person_schema = result["properties"]["person"]
    assert person_schema["type"] == "object"
    assert person_schema["additionalProperties"] is False
    assert "name" in person_schema["properties"]
    assert "age" in person_schema["properties"]
    assert "name" in person_schema["required"]
    assert "age" in person_schema["required"]


def test_array_schema():
    """Test array schema enforcement."""
    schema = {
        "type": "array",
        "items": {"type": "object", "properties": {"name": {"type": "string"}}},
    }

    result = ensure_strict_json_schema(schema)

    assert result["type"] == "array"
    assert "items" in result
    assert result["items"]["type"] == "object"
    assert result["items"]["additionalProperties"] is False
    assert "name" in result["items"]["properties"]
    assert "name" in result["items"]["required"]


def test_schema_with_definitions():
    """Test schema with definitions."""
    schema = {
        "definitions": {"Person": {"type": "object", "properties": {"name": {"type": "string"}}}},
        "type": "object",
        "properties": {"person": {"$ref": "#/definitions/Person"}},
    }

    result = ensure_strict_json_schema(schema)

    assert "definitions" in result
    assert "Person" in result["definitions"]
    assert result["definitions"]["Person"]["type"] == "object"
    assert result["definitions"]["Person"]["additionalProperties"] is False
    assert "name" in result["definitions"]["Person"]["properties"]
    assert "name" in result["definitions"]["Person"]["required"]


def test_schema_with_logical_operators():
    """Test schema with logical operators."""
    schema = {
        "type": "object",
        "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }

    result = ensure_strict_json_schema(schema)

    assert result["type"] == "object"
    assert "value" in result["properties"]
    assert "anyOf" in result["properties"]["value"]
    assert len(result["properties"]["value"]["anyOf"]) == 2
    assert result["properties"]["value"]["anyOf"][0]["type"] == "string"
    assert result["properties"]["value"]["anyOf"][1]["type"] == "integer"


def test_schema_with_allof_single():
    """Test schema with single allOf operator."""
    schema = {"type": "object", "properties": {"value": {"allOf": [{"type": "string"}]}}}

    result = ensure_strict_json_schema(schema)

    assert result["type"] == "object"
    assert "value" in result["properties"]
    assert "allOf" not in result["properties"]["value"]
    assert result["properties"]["value"]["type"] == "string"


def test_invalid_additional_properties():
    """Test schema with invalid additionalProperties."""
    schema = {"type": "object", "additionalProperties": True}

    with pytest.raises(UsageError):
        ensure_strict_json_schema(schema)


def test_resolve_schema_ref():
    """Test schema reference resolution."""
    schema = {
        "definitions": {"Person": {"type": "object", "properties": {"name": {"type": "string"}}}}
    }

    result = resolve_schema_ref(root=schema, ref="#/definitions/Person")

    assert result["type"] == "object"
    assert "name" in result["properties"]
    assert result["properties"]["name"]["type"] == "string"


def test_invalid_schema_ref():
    """Test invalid schema reference."""
    schema = {
        "definitions": {"Person": {"type": "object", "properties": {"name": {"type": "string"}}}}
    }

    with pytest.raises(ModelError):
        resolve_schema_ref(root=schema, ref="#/definitions/Invalid")


def test_invalid_ref_format():
    """Test invalid reference format."""
    schema = {
        "definitions": {"Person": {"type": "object", "properties": {"name": {"type": "string"}}}}
    }

    with pytest.raises(ModelError):
        resolve_schema_ref(root=schema, ref="invalid_ref")
