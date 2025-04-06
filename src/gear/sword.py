"""
Swords Module - Function Wrapping and Schema Generation

This module provides a powerful framework for creating and managing function-based swords,
which are specialized tools that wrap Python functions with enhanced capabilities including:
- Automatic schema generation from function signatures
- Docstring parsing and documentation extraction
- Type validation and JSON schema generation
- Context-aware function execution
- Error handling and reporting
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    Any,
    Concatenate,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from griffe import Docstring, DocstringSectionKind
from pydantic import BaseModel, Field, create_model
from typing_extensions import ParamSpec

from ..util._constants import LRU_CACHE_SIZE
from ..util._exceptions import UsageError
from ..util._items import RunItem
from ..util._strict_schema import ensure_strict_json_schema
from ..util._types import MaybeAwaitable, RunContextWrapper

########################################################
#               Private Types
########################################################

SwordParams = ParamSpec("SwordParams")
SwordFunctionWithoutContext = Callable[SwordParams, Any]
SwordFunctionWithContext = Callable[Concatenate[RunContextWrapper[Any], SwordParams], Any]
SwordFunction = SwordFunctionWithoutContext[SwordParams] | SwordFunctionWithContext[SwordParams]


########################################################
#       Data classes for Function Sword Schema
########################################################


@dataclass(frozen=True)
class FuncSchema:
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
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                positional_params.append(name)
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

        for name in self._positional_params:
            if hasattr(data, name):
                positional_args.append(getattr(data, name))

        if self._var_positional and hasattr(data, self._var_positional):
            var_args = getattr(data, self._var_positional)
            if var_args is not None:
                positional_args.extend(var_args)

        for name in self._keyword_params:
            if hasattr(data, name):
                keyword_args[name] = getattr(data, name)

        # Process variable keyword arguments (**kwargs)
        if self._var_keyword and hasattr(data, self._var_keyword):
            var_kwargs = getattr(data, self._var_keyword)
            if var_kwargs is not None:
                keyword_args.update(var_kwargs)

        return positional_args, keyword_args


@dataclass(frozen=True)
class FuncDocumentation:
    """Function metadata from docstring."""

    name: str
    description: str | None
    param_descriptions: dict[str, str] | None


@dataclass(frozen=True)
class SwordResult:
    """Result of running a sword."""

    sword: Sword
    output: Any
    run_item: RunItem


########################################################
#           Sword Types
########################################################


def default_sword_error_function(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """Default error handler for sword failures."""
    return f"An error occurred while running the sword. Error: {str(error)}"


SwordErrorFunction = Callable[[RunContextWrapper[Any], Exception], MaybeAwaitable[str]]


@dataclass(frozen=True)
class Sword:
    """Sword that wraps a Python function."""

    name: str
    description: str
    params_json_schema: dict[str, Any]
    on_invoke_sword: Callable[[RunContextWrapper[Any], str], Awaitable[Any]]
    strict_json_schema: bool = True
    failure_error_function: SwordErrorFunction | None = default_sword_error_function


########################################################
#               Private Methods
########################################################


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _detect_docstring_style(doc: str):
    """Detect docstring style using pattern matching."""
    patterns = {
        "sphinx": [
            re.compile(r"^:param\s", re.MULTILINE),
            re.compile(r"^:type\s", re.MULTILINE),
            re.compile(r"^:return:", re.MULTILINE),
            re.compile(r"^:rtype:", re.MULTILINE),
        ],
        "numpy": [
            re.compile(r"^Parameters\s*\n\s*-{3,}", re.MULTILINE),
            re.compile(r"^Returns\s*\n\s*-{3,}", re.MULTILINE),
            re.compile(r"^Yields\s*\n\s*-{3,}", re.MULTILINE),
        ],
        "google": [
            re.compile(r"^(Args|Arguments):", re.MULTILINE),
            re.compile(r"^(Returns):", re.MULTILINE),
            re.compile(r"^(Raises):", re.MULTILINE),
        ],
    }

    scores = {
        style: sum(1 for pattern in style_patterns if pattern.search(doc))
        for style, style_patterns in patterns.items()
    }

    return next(
        (style for style, score in scores.items() if score == max(scores.values())),
        "google",
    )


@contextlib.contextmanager
def _suppress_griffe_logging():
    """Suppress griffe warnings."""
    logger = logging.getLogger("griffe")
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _process_var_positional(
    param: inspect.Parameter, ann: Any, field_description: str | None
) -> tuple[Any, Field]:
    """Process *args parameters."""
    if get_origin(ann) is tuple:
        args_of_tuple = get_args(ann)
        if len(args_of_tuple) == 2 and args_of_tuple[1] is Ellipsis:
            ann = list[args_of_tuple[0]]  # type: ignore
        else:
            ann = list[Any]
    else:
        ann = list[ann]  # type: ignore

    return ann, Field(default_factory=list, description=field_description)  # type: ignore


@lru_cache(maxsize=LRU_CACHE_SIZE)
def _process_var_keyword(
    param: inspect.Parameter, ann: Any, field_description: str | None
) -> tuple[Any, Field]:
    """Process **kwargs parameters."""
    if get_origin(ann) is dict:
        dict_args = get_args(ann)
        if len(dict_args) == 2:
            ann = dict[dict_args[0], dict_args[1]]  # type: ignore
        else:
            ann = dict[str, Any]
    else:
        ann = dict[str, ann]  # type: ignore

    return ann, Field(default_factory=dict, description=field_description)  # type: ignore


########################################################
#               Public Methods
########################################################


def generate_func_documentation(func: Callable[..., Any], style: None) -> FuncDocumentation:
    """Extract function metadata from docstring."""
    doc = inspect.getdoc(func)
    if not doc:
        return FuncDocumentation(name=func.__name__, description=None, param_descriptions=None)

    with _suppress_griffe_logging():
        docstring = Docstring(doc, lineno=1, parser=style or _detect_docstring_style(doc))
        parsed = docstring.parse()

    description = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text),
        None,
    )

    param_descriptions = {
        param.name: param.description
        for section in parsed
        if section.kind == DocstringSectionKind.parameters
        for param in section.value
    }

    return FuncDocumentation(
        name=func.__name__,
        description=description,
        param_descriptions=param_descriptions or None,
    )


def function_schema(
    func: Callable[..., Any],
    docstring_style: None,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    """Extract function schema for sword use."""

    doc_info = generate_func_documentation(func, docstring_style) if use_docstring_info else None
    param_descs = doc_info.param_descriptions or {} if doc_info else {}
    func_name = name_override or doc_info.name if doc_info else func.__name__

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    params = list(sig.parameters.items())
    takes_context = False
    filtered_params = []

    if params:
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

    for name, param in params[1:]:
        ann = type_hints.get(name, param.annotation)
        if ann != inspect._empty:
            origin = get_origin(ann) or ann
            if origin is RunContextWrapper:
                raise UsageError(
                    f"RunContextWrapper param found at non-first position in {func.__name__}"
                )
        filtered_params.append((name, param))

    fields: dict[str, Any] = {}

    for name, param in filtered_params:
        ann = type_hints.get(name, param.annotation)
        default = param.default
        field_description = param_descs.get(name)

        if ann == inspect._empty:
            ann = Any

        if param.kind == param.VAR_POSITIONAL:
            ann, field = _process_var_positional(param, ann, field_description)
        elif param.kind == param.VAR_KEYWORD:
            ann, field = _process_var_keyword(param, ann, field_description)
        else:
            field = Field(
                ... if default == inspect._empty else default,
                description=field_description,
            )

        fields[name] = (ann, field)

    dynamic_model = create_model(f"{func_name}_args", __base__=BaseModel, **fields)
    json_schema = dynamic_model.model_json_schema()
    if strict_json_schema:
        json_schema = ensure_strict_json_schema(json_schema)

    return FuncSchema(
        name=func_name,
        description=description_override or doc_info.description if doc_info else None,
        params_pydantic_model=dynamic_model,
        params_json_schema=json_schema,
        signature=sig,
        on_invoke_sword=lambda ctx, sword_name: func(ctx.value, **ctx.args),
        takes_context=takes_context,
        strict_json_schema=strict_json_schema,
    )


########################################################
#           Sword Types
########################################################


@overload
def function_sword(
    func: SwordFunction[...],
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: None = None,
    use_docstring_info: bool = True,
    failure_error_function: SwordErrorFunction | None = None,
    strict_mode: bool = True,
) -> Sword: ...


@overload
def function_sword(
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: None = None,
    use_docstring_info: bool = True,
    failure_error_function: SwordErrorFunction | None = None,
    strict_mode: bool = True,
) -> Callable[[SwordFunction[...]], Sword]: ...


def function_sword(
    func: SwordFunction[...] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: None = None,
    use_docstring_info: bool = True,
    failure_error_function: SwordErrorFunction | None = default_sword_error_function,
    strict_mode: bool = True,
) -> Sword | Callable[[SwordFunction[...]], Sword]:
    def _create_function_sword(the_func: SwordFunction[...]) -> Sword:
        schema = function_schema(
            func=the_func,
            name_override=name_override,
            description_override=description_override,
            docstring_style=docstring_style,
            use_docstring_info=use_docstring_info,
            strict_json_schema=strict_mode,
        )

        async def _on_invoke_sword(ctx: RunContextWrapper[Any], input: str) -> Any:
            try:
                return await schema.on_invoke_sword(ctx, input)
            except Exception as e:
                if failure_error_function:
                    error_msg = failure_error_function(ctx, e)
                    if inspect.iscoroutine(error_msg):
                        error_msg = await error_msg
                    return error_msg
                raise

        return Sword(
            name=schema.name,
            description=schema.description,
            params_json_schema=schema.params_json_schema,
            on_invoke_sword=_on_invoke_sword,
            strict_json_schema=strict_mode,
            failure_error_function=failure_error_function,
        )

    def decorator(real_func: SwordFunction[...]) -> Sword:
        return _create_function_sword(real_func)

    if func is None:
        return decorator
    return decorator(func)
