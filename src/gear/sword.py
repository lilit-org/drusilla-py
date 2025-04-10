"""
Swords Module - Function Wrapping and Schema Generation

This module provides a framework for creating and managing function-based swords,
which are specialized tools that wrap Python functions with enhanced capabilities.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import (
    Any,
    Concatenate,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model
from typing_extensions import ParamSpec

from ..runners.items import RunItem
from ..util.constants import err
from ..util.exceptions import ModelError, UsageError, create_error_handler
from ..util.schema import ensure_strict_json_schema
from ..util.types import (
    MaybeAwaitable,
    RunContextWrapper,
)

#############################################################
#  Sword System's Private Types Safety and Functionalities
#############################################################

SwordParams = ParamSpec("SwordParams")
SwordFunctionWithoutContext = Callable[SwordParams, Any]
SwordFunctionWithContext = Callable[Concatenate[RunContextWrapper[Any], SwordParams], Any]
SwordFunction = SwordFunctionWithoutContext[SwordParams] | SwordFunctionWithContext[SwordParams]
SwordErrorFunction = Callable[[RunContextWrapper[Any], Exception], MaybeAwaitable[str]]
SWORD_ERROR_HANDLER = create_error_handler(err.SWORD_ERROR)


########################################################
#           Sword Main Dataclasses
########################################################


@dataclass(frozen=True)
class Sword:
    name: str
    description: str
    params_json_schema: dict[str, Any]
    on_invoke_sword: Callable[[RunContextWrapper[Any], str], Awaitable[Any]]
    strict_json_schema: bool = True
    failure_error_function: (
        Callable[[RunContextWrapper[Any], Exception], MaybeAwaitable[str]] | None
    ) = SWORD_ERROR_HANDLER


@dataclass(frozen=True)
class SwordResult:
    sword: Sword
    output: Any
    run_item: RunItem


########################################################
#     Create sword decorator factory
########################################################


def create_sword_decorator(
    sword_class: type[Sword],
    sync_func_type: type,
    async_func_type: type,
):
    def decorator(
        func: sync_func_type | async_func_type | None = None,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        use_docstring_info: bool = True,
        failure_error_function: SwordErrorFunction | None = SWORD_ERROR_HANDLER,
        strict_mode: bool = True,
    ) -> sword_class | Callable[[sync_func_type | async_func_type], sword_class]:
        def create_sword(f: sync_func_type | async_func_type) -> sword_class:
            schema = function_schema(
                func=f,
                name_override=name_override,
                description_override=description_override,
                use_docstring_info=use_docstring_info,
                strict_json_schema=strict_mode,
            )

            async def on_invoke(ctx: RunContextWrapper[Any], input: str) -> Any:
                try:
                    return await schema.on_invoke_sword(ctx, input)
                except Exception as e:
                    if failure_error_function:
                        error_msg = failure_error_function(ctx, e)
                        if inspect.iscoroutinefunction(failure_error_function):
                            error_msg = await error_msg
                        raise ModelError(err.MODEL_ERROR.format(error=error_msg)) from e
                    raise ModelError(err.MODEL_ERROR.format(error=str(e))) from e

            return sword_class(
                name=schema.name,
                description=schema.description,
                params_json_schema=schema.params_json_schema,
                on_invoke_sword=on_invoke,
                strict_json_schema=strict_mode,
                failure_error_function=failure_error_function,
            )

        return create_sword if func is None else create_sword(func)

    return decorator


# Type aliases for sword functions
SwordFuncSync = Callable[[RunContextWrapper[Any], str], Any]
SwordFuncAsync = Callable[[RunContextWrapper[Any], str], Awaitable[Any]]

function_sword = create_sword_decorator(
    Sword,
    SwordFuncSync,
    SwordFuncAsync,
)


########################################################
#  Dataclass for FuncSchema and function_schema method
########################################################


@dataclass(frozen=True)
class FuncSchema:
    """Schema for a function that can be used as a sword."""

    name: str
    description: str | None
    params_pydantic_model: type[BaseModel]
    params_json_schema: dict[str, Any]
    signature: inspect.Signature
    on_invoke_sword: Callable[[RunContextWrapper[Any], str], Awaitable[Any]]
    takes_context: bool = False
    strict_json_schema: bool = True
    _positional_params: list[str] = field(init=False)
    _keyword_params: list[str] = field(init=False)
    _var_positional: str | None = field(init=False)
    _var_keyword: str | None = field(init=False)

    def __post_init__(self) -> None:
        """Initialize parameter lists after object creation."""
        positional_params: list[str] = []
        keyword_params: list[str] = []
        var_positional: str | None = None
        var_keyword: str | None = None

        # Skip context parameter if present
        params = list(self.signature.parameters.items())
        if self.takes_context and params:
            params = params[1:]

        for name, param in params:
            if param.kind == param.VAR_POSITIONAL:
                var_positional = name
            elif param.kind == param.VAR_KEYWORD:
                var_keyword = name
            elif param.kind == param.POSITIONAL_ONLY:
                positional_params.append(name)
            elif param.kind == param.POSITIONAL_OR_KEYWORD:
                if param.default == param.empty:
                    positional_params.append(name)
                else:
                    keyword_params.append(name)
            else:
                keyword_params.append(name)

        object.__setattr__(self, "_positional_params", positional_params)
        object.__setattr__(self, "_keyword_params", keyword_params)
        object.__setattr__(self, "_var_positional", var_positional)
        object.__setattr__(self, "_var_keyword", var_keyword)

    def to_call_args(self, data: BaseModel) -> tuple[list[Any], dict[str, Any]]:
        """Convert Pydantic model to function call arguments."""
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}

        # Handle positional arguments
        for name in self._positional_params:
            if hasattr(data, name):
                positional_args.append(getattr(data, name))

        # Handle variable positional arguments (*args)
        if self._var_positional and hasattr(data, self._var_positional):
            var_args = getattr(data, self._var_positional)
            if var_args is not None:
                positional_args.extend(var_args)

        # Handle keyword arguments
        for name in self._keyword_params:
            if hasattr(data, name):
                keyword_args[name] = getattr(data, name)

        # Handle variable keyword arguments (**kwargs)
        if self._var_keyword and hasattr(data, self._var_keyword):
            var_kwargs = getattr(data, self._var_keyword)
            if var_kwargs is not None:
                keyword_args.update(var_kwargs)

        return positional_args, keyword_args


def function_schema(
    func: Callable[..., Any],
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    # Extract documentation and basic function info
    doc_info = generate_func_documentation(func) if use_docstring_info else None
    func_name = name_override or (doc_info.name if doc_info else func.__name__)
    param_descs = doc_info.param_descriptions or {} if doc_info else {}

    # Get function signature and type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Process parameters and detect context
    takes_context, filtered_params = _process_parameters(sig, type_hints)

    # Create Pydantic model fields
    fields = _create_pydantic_fields(filtered_params, type_hints, param_descs)

    # Create base model with desired configuration
    class DynamicBase(BaseModel):
        model_config = {"extra": "forbid" if strict_json_schema else "allow", "strict": True}

    # Create dynamic model and schema
    dynamic_model = create_model(
        f"{func_name}_args",
        __base__=DynamicBase,
        **fields,
    )

    json_schema = dynamic_model.model_json_schema()
    if strict_json_schema:
        json_schema = ensure_strict_json_schema(json_schema)

    # Create invocation handler
    on_invoke_sword = _create_invocation_handler(
        func,
        dynamic_model,
        sig,
        takes_context,
        strict_json_schema,
    )

    return FuncSchema(
        name=func_name,
        description=description_override or (doc_info.description if doc_info else None),
        params_pydantic_model=dynamic_model,
        params_json_schema=json_schema,
        signature=sig,
        on_invoke_sword=on_invoke_sword,
        takes_context=takes_context,
        strict_json_schema=strict_json_schema,
    )


########################################################
#             Invocation Methods
########################################################


def _create_invocation_handler(
    func: Callable[..., Any],
    dynamic_model: type[BaseModel],
    sig: inspect.Signature,
    takes_context: bool,
    strict_json_schema: bool = True,
) -> Callable[[RunContextWrapper[Any], str], Awaitable[Any]]:
    """Create the sword invocation handler."""

    async def on_invoke_sword(ctx: RunContextWrapper[Any], input: str) -> Any:
        try:
            json_data = json.loads(input)
            data = dynamic_model.model_validate(json_data, strict=strict_json_schema)

            args, kwargs = FuncSchema(
                name=func.__name__,
                description=None,
                params_pydantic_model=dynamic_model,
                params_json_schema=dynamic_model.model_json_schema(),
                signature=sig,
                on_invoke_sword=lambda _, __: None,
                takes_context=takes_context,
                strict_json_schema=strict_json_schema,
            ).to_call_args(data)

            if takes_context:
                args.insert(0, ctx)

            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        except Exception as e:
            raise ModelError(err.MODEL_ERROR.format(error=str(e))) from e

    return on_invoke_sword


def _process_parameters(
    sig: inspect.Signature, type_hints: dict[str, Any]
) -> tuple[bool, list[tuple[str, inspect.Parameter]]]:
    """Process function parameters and detect context parameter."""
    params = list(sig.parameters.items())
    takes_context = False
    filtered_params = []

    if not params:
        return False, []

    # Check first parameter for context
    first_name, first_param = params[0]
    ann = type_hints.get(first_name, first_param.annotation)
    if ann != inspect._empty:
        origin = get_origin(ann) or ann
        if origin is RunContextWrapper:
            takes_context = True
        else:
            filtered_params.append((first_name, first_param))
    else:
        filtered_params.append((first_name, first_param))

    # Process remaining parameters
    for name, param in params[1:]:
        ann = type_hints.get(name, param.annotation)
        if ann != inspect._empty:
            origin = get_origin(ann) or ann
            if origin is RunContextWrapper:
                raise UsageError(err.USAGE_ERROR.format(error=sig.name))
        filtered_params.append((name, param))

    return takes_context, filtered_params


########################################################
#        FuncDocumentation Dataclass and methods
########################################################


@dataclass(frozen=True)
class FuncDocumentation:
    """Function metadata from docstring."""

    name: str
    description: str | None
    param_descriptions: dict[str, str] | None


def generate_func_documentation(func: Callable[..., Any]) -> FuncDocumentation:
    """Extract function metadata from docstring."""
    doc = inspect.getdoc(func)
    if not doc:
        return FuncDocumentation(name=func.__name__, description=None, param_descriptions=None)

    # Simple docstring parsing that works with common formats
    lines = doc.strip().split("\n")
    description = []
    param_descriptions = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for parameter descriptions
        if line.startswith((":param", ":type", "Args:", "Parameters:")):
            continue
        elif ":" in line and not line.startswith(" "):
            # Assume this is a parameter description
            param, desc = line.split(":", 1)
            param_descriptions[param.strip()] = desc.strip()
        else:
            description.append(line)

    return FuncDocumentation(
        name=func.__name__,
        description="\n".join(description) if description else None,
        param_descriptions=param_descriptions or None,
    )


def _create_pydantic_fields(
    params: list[tuple[str, inspect.Parameter]],
    type_hints: dict[str, Any],
    param_descs: dict[str, str],
) -> dict[str, tuple[Any, Field]]:

    fields: dict[str, tuple[Any, Field]] = {}

    for name, param in params:
        ann = type_hints.get(name, param.annotation)
        default = param.default
        field_description = param_descs.get(name)

        if ann == inspect._empty:
            ann = Any

        if param.kind == param.VAR_POSITIONAL:
            # Handle *args - always use list[Any]
            ann = list[Any]
            field = Field(default_factory=list, description=field_description)
        elif param.kind == param.VAR_KEYWORD:
            # Handle **kwargs - always use dict[str, Any]
            ann = dict[str, Any]
            field = Field(default_factory=dict, description=field_description)
        else:
            field = Field(
                ... if default == inspect._empty else default,
                description=field_description,
            )

        fields[name] = (ann, field)

    return fields
