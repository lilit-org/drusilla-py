"""Tests for the sword module."""

import inspect
from typing import Any

import pytest
from pydantic import BaseModel

from src.gear.sword import (
    FuncSchema,
    Sword,
    function_sword,
)
from src.util._constants import ERROR_MESSAGES
from src.util._exceptions import create_error_handler
from src.util._types import RunContextWrapper


class SwordTestInput(BaseModel):
    """Test input model for sword."""

    message: str
    count: int = 1


@pytest.fixture
def mock_context() -> RunContextWrapper[Any]:
    """Fixture for creating a mock context."""
    return RunContextWrapper(context={})


class TestSwordErrorHandling:
    """Test suite for sword error handling functionality."""

    def test_error_handler(self, mock_context: RunContextWrapper[Any]):
        """Test the sword error handler function."""
        error = Exception("Test error")
        error_handler = create_error_handler(ERROR_MESSAGES.SWORD_ERROR.message)
        result = error_handler(mock_context, error)
        assert result == ERROR_MESSAGES.SWORD_ERROR.message.format(error="Test error")

    @pytest.mark.asyncio
    async def test_function_sword_with_error_handling(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with error handling."""

        @function_sword(failure_error_function=None)
        async def test_sword(message: str) -> str:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')


class TestFuncSchema:
    """Test suite for FuncSchema functionality."""

    def test_to_call_args(self):
        """Test converting Pydantic model to function call arguments."""

        class TestModel(BaseModel):
            a: int
            b: str
            c: list[int] = []
            d: dict[str, Any] = {}

        def test_func(
            a: int, b: str, c: list[int] | None = None, d: dict[str, Any] | None = None
        ) -> None:
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


class TestFunctionSword:
    """Test suite for function_sword decorator functionality."""

    @pytest.mark.asyncio
    async def test_basic_functionality(self, mock_context: RunContextWrapper[Any]):
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

    def test_custom_name_and_description(self):
        """Test function sword with custom name and description."""

        @function_sword(name_override="custom_sword", description_override="Custom description")
        async def test_sword(ctx: RunContextWrapper[Any], message: str) -> str:
            return message

        assert test_sword.name == "custom_sword"
        assert test_sword.description == "Custom description"

    @pytest.mark.asyncio
    async def test_var_positional_args(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with variable positional arguments."""

        @function_sword
        async def test_sword(*args: int) -> int:
            return sum(args)

        result = await test_sword.on_invoke_sword(mock_context, '{"args": [1, 2, 3]}')
        assert result == 6

    @pytest.mark.asyncio
    async def test_var_keyword_args(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with variable keyword arguments."""

        @function_sword(strict_mode=False)
        async def test_sword(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        result = await test_sword.on_invoke_sword(mock_context, '{"kwargs": {"a": 1, "b": 2}}')
        assert result == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_context_parameter(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with context parameter."""

        @function_sword
        async def test_sword(ctx: RunContextWrapper[Any], message: str) -> str:
            return f"Context: {message}"

        result = await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')
        assert result == "Context: test"
