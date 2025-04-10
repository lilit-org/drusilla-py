"""Tests for the sword module."""

import inspect
from typing import Any, get_type_hints

import pytest
from pydantic import BaseModel

from src.gear.sword import (
    FuncSchema,
    Sword,
    SwordResult,
    _create_pydantic_fields,
    _process_parameters,
    function_schema,
    function_sword,
    generate_func_documentation,
)
from src.runners.items import SwordCallOutputItem
from src.util.constants import err
from src.util.exceptions import ModelError, create_error_handler
from src.util.types import RunContextWrapper


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
        error_handler = create_error_handler(err.SWORD_ERROR)
        result = error_handler(mock_context, error)
        assert result == err.SWORD_ERROR.format(error="Test error")

    @pytest.mark.asyncio
    async def test_function_sword_error_handling(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with various error handling scenarios."""

        # Test with ValueError
        @function_sword
        async def test_sword(message: str) -> str:
            raise ValueError("Test error")

        with pytest.raises(ModelError):
            await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')

        # Test with invalid JSON
        @function_sword
        async def test_sword2(message: str) -> str:
            return message

        with pytest.raises(ModelError):
            await test_sword2.on_invoke_sword(mock_context, "invalid json")

        # Test with custom error handler
        def custom_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
            return f"Custom error: {str(error)}"

        @function_sword(failure_error_function=custom_error_handler)
        async def test_sword3(message: str) -> str:
            raise ValueError("Test error")

        with pytest.raises(ModelError):
            await test_sword3.on_invoke_sword(mock_context, '{"message": "test"}')

        # Test with async error handler
        async def async_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
            return f"Async error: {str(error)}"

        @function_sword(failure_error_function=async_error_handler)
        async def test_sword4(message: str) -> str:
            raise ValueError("Test error")

        with pytest.raises(ModelError):
            await test_sword4.on_invoke_sword(mock_context, '{"message": "test"}')


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

    def test_custom_metadata(self):
        """Test function sword with custom metadata."""

        # Test with custom name and description
        @function_sword(name_override="custom_sword", description_override="Custom description")
        async def test_sword(ctx: RunContextWrapper[Any], message: str) -> str:
            return message

        assert test_sword.name == "custom_sword"
        assert test_sword.description == "Custom description"

        # Test with docstring-based metadata
        @function_sword(use_docstring_info=True)
        async def test_sword2(ctx: RunContextWrapper[Any], message: str) -> str:
            """Test sword with docstring.

            This is a test sword that demonstrates docstring usage.
            """
            return message

        assert test_sword2.name == "test_sword2"
        assert "This is a test sword" in test_sword2.description

    @pytest.mark.asyncio
    async def test_parameter_handling(self, mock_context: RunContextWrapper[Any]):
        """Test function sword with various parameter types."""

        # Test with variable positional arguments
        @function_sword
        async def test_sword(*args: int) -> int:
            return sum(args)

        result = await test_sword.on_invoke_sword(mock_context, '{"args": [1, 2, 3]}')
        assert result == 6

        # Test with variable keyword arguments
        @function_sword(strict_mode=False)
        async def test_sword2(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        result = await test_sword2.on_invoke_sword(mock_context, '{"kwargs": {"a": 1, "b": 2}}')
        assert result == {"a": 1, "b": 2}

        # Test with context parameter
        @function_sword
        async def test_sword3(ctx: RunContextWrapper[Any], message: str) -> str:
            return f"Context: {message}"

        result = await test_sword3.on_invoke_sword(mock_context, '{"message": "test"}')
        assert result == "Context: test"

        # Test with synchronous function
        @function_sword
        def test_sword4(message: str) -> str:
            return message

        result = await test_sword4.on_invoke_sword(mock_context, '{"message": "test"}')
        assert result == "test"

    @pytest.mark.asyncio
    async def test_validation(self, mock_context: RunContextWrapper[Any]):
        """Test function sword validation."""

        @function_sword(strict_mode=True)
        async def test_sword(message: str) -> str:
            return message

        # Test with valid input
        result = await test_sword.on_invoke_sword(mock_context, '{"message": "test"}')
        assert result == "test"

        # Test with invalid input (extra field)
        with pytest.raises(ModelError):
            await test_sword.on_invoke_sword(mock_context, '{"message": "test", "extra": "field"}')

        # Test with invalid input (wrong type)
        @function_sword
        async def test_sword2(message: str, count: int) -> str:
            return f"{message} {count}"

        with pytest.raises(ModelError):
            await test_sword2.on_invoke_sword(
                mock_context, '{"message": "test", "count": "not a number"}'
            )

        # Test with missing required parameter
        with pytest.raises(ModelError):
            await test_sword2.on_invoke_sword(mock_context, '{"message": "test"}')


class TestSwordResult:
    """Test suite for SwordResult functionality."""

    @pytest.mark.asyncio
    async def test_sword_result_creation(self, mock_context: RunContextWrapper[Any]):
        """Test creating a SwordResult instance."""

        @function_sword
        async def test_sword(message: str) -> str:
            return message

        input_json = '{"message": "test"}'
        result = await test_sword.on_invoke_sword(mock_context, input_json)
        run_item = SwordCallOutputItem(
            output=result,
            raw_item={
                "type": "function_call_output",
                "call_id": "test",
                "output": result,
                "input": input_json,
            },
            agent=None,
        )
        sword_result = SwordResult(sword=test_sword, output=result, run_item=run_item)

        assert sword_result.sword == test_sword
        assert sword_result.output == "test"
        assert isinstance(sword_result.run_item, SwordCallOutputItem)
        assert sword_result.run_item.output == "test"
        assert sword_result.run_item.raw_item["output"] == "test"


class TestProcessParameters:
    """Test suite for _process_parameters functionality."""

    def test_process_parameters_with_context(self):
        """Test processing parameters with context parameter."""

        def test_func(ctx: RunContextWrapper[Any], message: str) -> None:
            pass

        type_hints = get_type_hints(test_func)
        sig = inspect.signature(test_func)
        takes_context, params = _process_parameters(sig, type_hints)

        assert takes_context is True
        assert len(params) == 1
        assert params[0][0] == "message"

    def test_process_parameters_without_context(self):
        """Test processing parameters without context parameter."""

        def test_func(message: str) -> None:
            pass

        type_hints = get_type_hints(test_func)
        sig = inspect.signature(test_func)
        takes_context, params = _process_parameters(sig, type_hints)

        assert takes_context is False
        assert len(params) == 1
        assert params[0][0] == "message"

    def test_process_parameters_with_complex_types(self):
        """Test processing parameters with complex type hints."""

        def test_func(
            message: str,
            items: list[int],
            config: dict[str, Any],
            optional: str | None = None,
        ) -> None:
            pass

        type_hints = get_type_hints(test_func)
        sig = inspect.signature(test_func)
        takes_context, params = _process_parameters(sig, type_hints)

        assert takes_context is False
        assert len(params) == 4
        assert all(name in ["message", "items", "config", "optional"] for name, _ in params)


class TestCreatePydanticFields:
    """Test suite for _create_pydantic_fields functionality."""

    def test_create_pydantic_fields_basic(self):
        """Test creating basic Pydantic fields."""

        def test_func(message: str, count: int = 1) -> None:
            pass

        sig = inspect.signature(test_func)
        params = list(sig.parameters.items())
        type_hints = get_type_hints(test_func)
        param_descs = {"message": "The message to process"}

        fields = _create_pydantic_fields(params, type_hints, param_descs)

        assert "message" in fields
        assert "count" in fields
        assert fields["message"][1].description == "The message to process"
        assert fields["count"][1].default == 1

    def test_create_pydantic_fields_with_complex_types(self):
        """Test creating Pydantic fields with complex types."""

        def test_func(
            items: list[int], config: dict[str, Any], optional: str | None = None
        ) -> None:
            pass

        sig = inspect.signature(test_func)
        params = list(sig.parameters.items())
        type_hints = get_type_hints(test_func)
        param_descs = {}

        fields = _create_pydantic_fields(params, type_hints, param_descs)

        assert "items" in fields
        assert "config" in fields
        assert "optional" in fields
        assert fields["optional"][1].default is None


class TestGenerateFuncDocumentation:
    """Test suite for generate_func_documentation functionality."""

    def test_generate_func_documentation_with_docstring(self):
        """Test generating function documentation from docstring."""

        def test_func(message: str) -> str:
            """Test function with docstring.

            message: The message to process
            """
            return message

        doc_info = generate_func_documentation(test_func)

        assert doc_info.name == "test_func"
        assert doc_info.description == "Test function with docstring."
        assert doc_info.param_descriptions == {"message": "The message to process"}

    def test_generate_func_documentation_without_docstring(self):
        """Test generating function documentation without docstring."""

        def test_func(message: str) -> str:
            return message

        doc_info = generate_func_documentation(test_func)

        assert doc_info.name == "test_func"
        assert doc_info.description is None
        assert doc_info.param_descriptions is None

    def test_generate_func_documentation_with_partial_docstring(self):
        """Test generating function documentation with partial docstring."""

        def test_func(message: str) -> str:
            """Test function with partial docstring."""
            return message

        doc_info = generate_func_documentation(test_func)

        assert doc_info.name == "test_func"
        assert doc_info.description == "Test function with partial docstring."
        assert doc_info.param_descriptions is None


class TestFunctionSchema:
    """Test suite for function_schema functionality."""

    def test_function_schema_with_invalid_function(self):
        """Test function_schema with invalid function."""
        with pytest.raises(AttributeError) as exc_info:
            function_schema(None)
        assert "'NoneType' object has no attribute '__name__'" in str(exc_info.value)

    def test_function_schema_with_custom_name_and_description(self):
        """Test function_schema with custom name and description."""

        def test_func(message: str) -> str:
            return message

        schema = function_schema(
            test_func,
            name_override="custom_name",
            description_override="Custom description",
            use_docstring_info=False,
        )

        assert schema.name == "custom_name"
        assert schema.description == "Custom description"

    def test_function_schema_with_strict_mode(self):
        """Test function_schema with strict mode enabled."""

        def test_func(message: str) -> str:
            return message

        schema = function_schema(test_func, strict_json_schema=True)
        assert schema.strict_json_schema is True

        schema = function_schema(test_func, strict_json_schema=False)
        assert schema.strict_json_schema is False
