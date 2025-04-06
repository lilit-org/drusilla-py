"""Unit tests for the AgentOutputSchema module."""

import pytest
from pydantic import BaseModel

from src.agents.output import AgentOutputSchema
from src.util._exceptions import ModelError, UsageError


# Test fixtures
@pytest.fixture
def test_model():
    class TestModel(BaseModel):
        name: str
        age: int | None = None
        is_active: bool = True

    return TestModel


# Test cases
def test_plain_text_schema():
    """Test schema creation for plain text output."""
    schema = AgentOutputSchema(str)
    assert schema.is_plain_text() is True
    assert schema.output_type is str
    assert schema.strict_json_schema is True


def test_model_schema(test_model):
    """Test schema creation for Pydantic model output."""
    schema = AgentOutputSchema(test_model)
    assert schema.is_plain_text() is False
    assert schema.output_type is test_model
    assert schema.strict_json_schema is True

    json_schema = schema.json_schema()
    assert "properties" in json_schema
    assert "name" in json_schema["properties"]
    assert "age" in json_schema["properties"]
    assert "is_active" in json_schema["properties"]


def test_plain_text_json_schema_error():
    """Test that getting JSON schema for plain text raises error."""
    schema = AgentOutputSchema(str)
    with pytest.raises(UsageError):
        schema.json_schema()


def test_validate_json_plain_text():
    """Test JSON validation for plain text output."""
    schema = AgentOutputSchema(str)
    result = schema.validate_json('"test string"')
    assert result == "test string"


def test_validate_json_model(test_model):
    """Test JSON validation for Pydantic model output."""
    schema = AgentOutputSchema(test_model)
    json_str = '{"name": "John", "age": 30}'
    result = schema.validate_json(json_str)
    assert isinstance(result, test_model)
    assert result.name == "John"
    assert result.age == 30
    assert result.is_active is True  # Default value


def test_validate_json_partial(test_model):
    """Test partial JSON validation."""
    schema = AgentOutputSchema(test_model)

    # Test with invalid data (missing required field)
    json_str = '{"age": 30}'
    with pytest.raises(ModelError):  # Should fail as name is required
        schema.validate_json(json_str)

    # Test with valid data
    json_str = '{"name": "John", "age": 30}'
    result = schema.validate_json(json_str)
    assert result.name == "John"
    assert result.age == 30

    # Test with partial data (incomplete JSON)
    json_str = '{"name": "John", "age": 30'  # Missing closing brace
    with pytest.raises(ModelError):  # Should fail with standard validation
        schema.validate_json(json_str)

    # Test with partial validation - should handle incomplete JSON
    result = schema.validate_json(json_str, partial=True)
    assert result.name == "John"
    assert result.age == 30


def test_output_type_name():
    """Test getting output type name."""
    schema = AgentOutputSchema(str)
    assert schema.output_type_name() == "str"


def test_non_strict_schema(test_model):
    """Test schema creation with strict validation disabled."""
    schema = AgentOutputSchema(test_model, strict_json_schema=False)
    assert schema.strict_json_schema is False
    json_schema = schema.json_schema()
    assert "properties" in json_schema
