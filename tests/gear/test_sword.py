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
from src.util._exceptions import ModelError, create_error_handler
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

        @function_sword
        async def test_sword(message: str) -> str:
            raise ValueError("Test error")

        with pytest.raises(ModelError) as exc_info:
            await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')
        assert isinstance(exc_info.value, ModelError)

    @pytest.mark.asyncio
    async def test_function_sword_with_invalid_json(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with invalid JSON input."""

        @function_sword
        async def test_sword(message: str) -> str:
            return message

        with pytest.raises(ModelError) as exc_info:
            await test_sword.on_invoke_sword(mock_context, "invalid json")
        assert isinstance(exc_info.value, ModelError)

    @pytest.mark.asyncio
    async def test_function_sword_with_custom_error_handler(
        self, mock_context: RunContextWrapper[Any]
    ):
        """Test function sword with custom error handler."""

        def custom_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
            return f"Custom error: {str(error)}"

        @function_sword(failure_error_function=custom_error_handler)
        async def test_sword(message: str) -> str:
            raise ValueError("Test error")

        with pytest.raises(ModelError) as exc_info:
            await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')
        assert isinstance(exc_info.value, ModelError)

    @pytest.mark.asyncio
    async def test_function_sword_with_async_error_handler(
        self, mock_context: RunContextWrapper[Any]
    ):
        """Test function sword with async error handler."""

        async def custom_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
            return f"Async error: {str(error)}"

        @function_sword(failure_error_function=custom_error_handler)
        async def test_sword(message: str) -> str:
            raise ValueError("Test error")

        with pytest.raises(ModelError) as exc_info:
            await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')
        assert isinstance(exc_info.value, ModelError)


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

    def test_to_call_args_with_context(self):
        """Test converting Pydantic model to function call arguments with context."""

        class TestModel(BaseModel):
            message: str

        def test_func(ctx: RunContextWrapper[Any], message: str) -> None:
            pass

        schema = FuncSchema(
            name="test",
            description="Test schema",
            params_pydantic_model=TestModel,
            params_json_schema=TestModel.model_json_schema(),
            signature=inspect.signature(test_func),
            on_invoke_sword=lambda ctx, input: None,
            takes_context=True,
            strict_json_schema=True,
        )

        data = TestModel(message="test")
        positional_args, keyword_args = schema.to_call_args(data)

        assert positional_args == ["test"]
        assert keyword_args == {}

    def test_to_call_args_with_var_positional(self):
        """Test converting Pydantic model with variable positional arguments."""

        class TestModel(BaseModel):
            args: list[int]

        def test_func(*args: int) -> None:
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

        data = TestModel(args=[1, 2, 3])
        positional_args, keyword_args = schema.to_call_args(data)

        assert positional_args == [1, 2, 3]
        assert keyword_args == {}

    def test_to_call_args_with_var_keyword(self):
        """Test converting Pydantic model with variable keyword arguments."""

        class TestModel(BaseModel):
            kwargs: dict[str, Any]

        def test_func(**kwargs: Any) -> None:
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

        data = TestModel(kwargs={"a": 1, "b": 2})
        positional_args, keyword_args = schema.to_call_args(data)

        assert positional_args == []
        assert keyword_args == {"a": 1, "b": 2}


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

    def test_docstring_name_and_description(self):
        """Test function sword with docstring-based name and description."""

        @function_sword(use_docstring_info=True)
        async def test_sword(ctx: RunContextWrapper[Any], message: str) -> str:
            """Test sword with docstring.

            This is a test sword that demonstrates docstring usage.
            """
            return message

        assert test_sword.name == "test_sword"
        assert "This is a test sword" in test_sword.description

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

    @pytest.mark.asyncio
    async def test_sync_function(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with synchronous function."""

        @function_sword
        def test_sword(message: str) -> str:
            return message

        result = await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')
        assert result == "test"

    @pytest.mark.asyncio
    async def test_strict_json_schema(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with strict JSON schema validation."""

        @function_sword(strict_mode=True)
        async def test_sword(message: str) -> str:
            return message

        # Test with valid input
        result = await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')
        assert result == "test"

        # Test with invalid input (extra field)
        with pytest.raises(ModelError):
            await test_sword.on_invoke_sword(mock_context, '{"message": "test", "extra": "field"}')

    @pytest.mark.asyncio
    async def test_parameter_validation(self, mock_context: RunContextWrapper[Any]):
        """Test function sword parameter validation."""

        @function_sword
        async def test_sword(message: str, count: int) -> str:
            return f"{message} {count}"

        # Test with valid input
        result = await test_sword.on_invoke_sword(mock_context, '{"message": "test", "count": 1}')
        assert result == "test 1"

        # Test with invalid input (wrong type)
        with pytest.raises(ModelError):
            await test_sword.on_invoke_sword(
                mock_context, '{"message": "test", "count": "not a number"}'
            )

        # Test with missing required parameter
        with pytest.raises(ModelError):
            await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')

    @pytest.mark.asyncio
    async def test_nested_models(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with nested Pydantic models."""

        class NestedModel(BaseModel):
            value: str

        class TestModel(BaseModel):
            nested: NestedModel

        @function_sword
        async def test_sword(nested: NestedModel) -> str:
            return nested.value

        result = await test_sword.on_invoke_sword(mock_context, '{"nested": {"value": "test"}}')
        assert result == "test"

    @pytest.mark.asyncio
    async def test_optional_parameters(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with optional parameters."""

        @function_sword
        async def test_sword(message: str, optional: str | None = None) -> str:
            return f"{message} {optional}" if optional else message

        # Test without optional parameter
        result = await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')
        assert result == "test"

        # Test with optional parameter
        result = await test_sword.on_invoke_sword(
            mock_context, '{"message": "test", "optional": "extra"}'
        )
        assert result == "test extra"

    @pytest.mark.asyncio
    async def test_invalid_json(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with invalid JSON input."""

        # Create a sword with strict mode enabled and no error handler
        @function_sword(strict_mode=True, failure_error_function=None)
        async def test_sword(message: str) -> str:
            return message

        # Test with malformed JSON
        with pytest.raises(ModelError):
            await test_sword.on_invoke_sword(mock_context, "{invalid json")

        # Test with empty string
        with pytest.raises(ModelError) as exc_info:
            await test_sword.on_invoke_sword(mock_context, "")
        assert "Expecting value" in str(exc_info.value)
