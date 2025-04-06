"""Tests for the sword module."""

import inspect
from typing import Any

import pytest
from pydantic import BaseModel

from src.gear.sword import (
    FuncSchema,
    Sword,
    default_sword_error_function,
    function_sword,
)
from src.util._types import RunContextWrapper


class SwordTestInput(BaseModel):
    """Test input model for sword."""

    message: str
    count: int = 1


@pytest.fixture
def mock_context() -> RunContextWrapper[Any]:
    """Fixture for creating a mock context."""
    return RunContextWrapper(context={})


def test_default_sword_error_function(mock_context: RunContextWrapper[Any]):
    """Test the default sword error function."""
    error = Exception("Test error")
    result = default_sword_error_function(mock_context, error)
    assert result == "An error occurred while running the sword. Error: Test error"


def test_func_schema_to_call_args():
    """Test converting Pydantic model to function call arguments."""

    class TestModel(BaseModel):
        a: int
        b: str
        c: list[int] = []
        d: dict[str, Any] = {}

    def test_func(a: int, b: str, c: list[int] = None, d: dict[str, Any] = None) -> None:
        pass

    schema = FuncSchema(
        name="test",
        description="Test schema",
        params_pydantic_model=TestModel,
        params_json_schema=TestModel.model_json_schema(),
        signature=inspect.signature(test_func),
        on_invoke_sword=lambda ctx, input: None,
        takes_context=False,
        strict_json_schema=True,
    )

    data = TestModel(a=1, b="test", c=[1, 2, 3], d={"key": "value"})
    positional_args, keyword_args = schema.to_call_args(data)

    assert positional_args == [1, "test"]
    assert keyword_args == {"c": [1, 2, 3], "d": {"key": "value"}}


@pytest.mark.asyncio
async def test_function_sword_basic(mock_context: RunContextWrapper[Any]):
    """Test basic function sword creation and invocation."""

    @function_sword
    async def test_sword(message: str, count: int = 1) -> str:
        return f"{message} {count}"

    assert isinstance(test_sword, Sword)
    assert test_sword.name == "test_sword"
    assert "message" in test_sword.params_json_schema["properties"]
    assert "count" in test_sword.params_json_schema["properties"]

    result = await test_sword.on_invoke_sword(mock_context, '{"message": "hello", "count": 2}')
    assert result == "hello 2"


@pytest.mark.asyncio
async def test_function_sword_with_docstring(mock_context: RunContextWrapper[Any]):
    """Test function sword with docstring."""

    @function_sword
    async def test_sword(ctx: RunContextWrapper[Any], message: str) -> str:
        """Test sword with docstring.

        Args:
            message: The message to process
        """
        return message.upper()

    assert isinstance(test_sword, Sword)
    assert test_sword.description == "Test sword with docstring."
    assert "message" in test_sword.params_json_schema["properties"]


@pytest.mark.asyncio
async def test_function_sword_with_error_handling(mock_context: RunContextWrapper[Any]):
    """Test function sword with error handling."""

    @function_sword(failure_error_function=None)
    async def test_sword(message: str) -> str:
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')


def test_function_sword_with_custom_name():
    """Test function sword with custom name."""

    @function_sword(name_override="custom_sword")
    async def test_sword(ctx: RunContextWrapper[Any], message: str) -> str:
        return message

    assert test_sword.name == "custom_sword"


def test_function_sword_with_custom_description():
    """Test function sword with custom description."""

    @function_sword(description_override="Custom description")
    async def test_sword(ctx: RunContextWrapper[Any], message: str) -> str:
        return message

    assert test_sword.description == "Custom description"


@pytest.mark.asyncio
async def test_function_sword_with_var_positional(mock_context: RunContextWrapper[Any]):
    """Test function sword with variable positional arguments."""

    @function_sword
    async def test_sword(*args: int) -> int:
        return sum(args)

    result = await test_sword.on_invoke_sword(mock_context, '{"args": [1, 2, 3]}')
    assert result == 6


@pytest.mark.asyncio
async def test_function_sword_with_var_keyword(mock_context: RunContextWrapper[Any]):
    """Test function sword with variable keyword arguments."""

    @function_sword(strict_mode=False)
    async def test_sword(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    result = await test_sword.on_invoke_sword(mock_context, '{"kwargs": {"a": 1, "b": 2}}')
    assert result == {"a": 1, "b": 2}
