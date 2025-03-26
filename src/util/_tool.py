from __future__ import annotations

import contextlib
import inspect
import json
import re
from collections.abc import Awaitable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints, overload

from griffe import Docstring, DocstringSectionKind
from pydantic import BaseModel, Field, ValidationError, create_model
from typing_extensions import Concatenate, ParamSpec

from ._computer import AsyncComputer, Computer
from ._exceptions import ModelError, UsageError
from ._items import RunItem
from ._logger import logger
from ._run_context import RunContextWrapper
from ._strict_schema import ensure_strict_json_schema
from ._types import MaybeAwaitable

########################################################
#               Private Types                           #
########################################################

ToolParams = ParamSpec("ToolParams")
ToolFunctionWithoutContext = Callable[ToolParams, Any]
ToolFunctionWithContext = Callable[Concatenate[RunContextWrapper[Any], ToolParams], Any]
ToolFunction = Union[ToolFunctionWithoutContext[ToolParams], ToolFunctionWithContext[ToolParams]]


########################################################
#               Data classes                           #
########################################################

@dataclass(frozen=True)
class FuncSchema:
    """Schema for Python functions used as LLM tools."""

    name: str
    description: str | None
    params_pydantic_model: type[BaseModel]
    params_json_schema: dict[str, Any]
    signature: inspect.Signature
    takes_context: bool = False
    strict_json_schema: bool = True
    _positional_params: list[str] = field(init=False)
    _keyword_params: list[str] = field(init=False)
    _var_positional: str | None = field(init=False)
    _var_keyword: str | None = field(init=False)

    def __post_init__(self) -> None:
        """Pre-compute parameter order and types."""
        positional_params: list[str] = []
        keyword_params: list[str] = []
        var_positional: str | None = None
        var_keyword: str | None = None

        for name, param in self.signature.parameters.items():
            if self.takes_context and name == list(self.signature.parameters.keys())[0]:
                continue

            if param.kind == param.VAR_POSITIONAL:
                var_positional = name
            elif param.kind == param.VAR_KEYWORD:
                var_keyword = name
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                positional_params.append(name)
            else:
                keyword_params.append(name)

        object.__setattr__(self, '_positional_params', positional_params)
        object.__setattr__(self, '_keyword_params', keyword_params)
        object.__setattr__(self, '_var_positional', var_positional)
        object.__setattr__(self, '_var_keyword', var_keyword)

    def to_call_args(self, data: BaseModel) -> tuple[list[Any], dict[str, Any]]:
        """Convert Pydantic model to function call arguments."""
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}

        for name in self._positional_params:
            value = getattr(data, name, None)
            positional_args.append(value)

        if self._var_positional:
            value = getattr(data, self._var_positional, None)
            if value:
                positional_args.extend(value)

        for name in self._keyword_params:
            value = getattr(data, name, None)
            keyword_args[name] = value

        if self._var_keyword:
            value = getattr(data, self._var_keyword, None)
            if value:
                keyword_args.update(value)

        return positional_args, keyword_args

@dataclass(frozen=True)
class FuncDocumentation:
    """Function metadata from docstring."""
    name: str
    description: str | None
    param_descriptions: dict[str, str] | None

@dataclass(frozen=True)
class FunctionToolResult:
    """Result of running a function tool."""
    tool: FunctionTool
    output: Any
    run_item: RunItem

@dataclass(frozen=True)
class FunctionTool:
    """Tool that wraps a Python function."""
    name: str
    description: str
    params_json_schema: dict[str, Any]
    on_invoke_tool: Callable[[RunContextWrapper[Any], str], Awaitable[Any]]
    strict_json_schema: bool = True

@dataclass(frozen=True)
class FileSearchTool:
    """Tool for searching vector stores."""
    vector_store_ids: list[str]
    max_num_results: int | None = None
    include_search_results: bool = False
    ranking_options: Any | None = None
    filters: Any | None = None

    @property
    def name(self):
        return "file_search"

@dataclass(frozen=True)
class WebSearchTool:
    """Tool for web searching."""
    user_location: Any | None = None
    search_context_size: Literal["low", "medium", "high"] = "medium"

    @property
    def name(self):
        return "web_search_preview"

@dataclass(frozen=True)
class ComputerTool:
    """Tool for computer control."""
    computer: Computer | AsyncComputer

    @property
    def name(self):
        return "computer_use_preview"


########################################################
#               Private Functions                       #
########################################################

DocstringStyle = Literal["google", "numpy", "sphinx"]
@lru_cache(maxsize=128)
def _detect_docstring_style(doc: str) -> DocstringStyle:
    """Detect docstring style using pattern matching."""
    scores: dict[DocstringStyle, int] = {"sphinx": 0, "numpy": 0, "google": 0}

    # Sphinx style: :param, :type, :return:, :rtype:
    sphinx_patterns = [r"^:param\s", r"^:type\s", r"^:return:", r"^:rtype:"]
    for pattern in sphinx_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["sphinx"] += 1

    # Numpy style: Headers with dashed underlines
    numpy_patterns = [
        r"^Parameters\s*\n\s*-{3,}",
        r"^Returns\s*\n\s*-{3,}",
        r"^Yields\s*\n\s*-{3,}",
    ]
    for pattern in numpy_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["numpy"] += 1

    # Google style: Section headers with colons
    google_patterns = [r"^(Args|Arguments):", r"^(Returns):", r"^(Raises):"]
    for pattern in google_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["google"] += 1

    max_score = max(scores.values())
    if max_score == 0:
        return "google"

    return next((style for style in ["sphinx", "numpy", "google"] if scores[style] == max_score), "google")

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

@lru_cache(maxsize=128)
def _process_var_positional(param: inspect.Parameter, ann: Any, field_description: str | None) -> tuple[Any, Field]:
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

@lru_cache(maxsize=128)
def _process_var_keyword(param: inspect.Parameter, ann: Any, field_description: str | None) -> tuple[Any, Field]:
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
#               Public Functions                        #
########################################################

def generate_func_documentation(
    func: Callable[..., Any], style: DocstringStyle | None = None
) -> FuncDocumentation:
    """Extract function metadata from docstring."""
    doc = inspect.getdoc(func)
    if not doc:
        return FuncDocumentation(name=func.__name__, description=None, param_descriptions=None)

    with _suppress_griffe_logging():
        docstring = Docstring(doc, lineno=1, parser=style or _detect_docstring_style(doc))
        parsed = docstring.parse()

    description = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text), None
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
    docstring_style: DocstringStyle | None = None,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    """Extract function schema for tool use."""

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
                    f"RunContextWrapper param found at non-first position in function {func.__name__}"
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
        takes_context=takes_context,
        strict_json_schema=strict_json_schema,
    )

Tool = Union[FunctionTool, FileSearchTool, WebSearchTool, ComputerTool]
"""A tool that can be used in an agent."""

def default_tool_error_function(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """Default error handler for tool failures."""
    return f"An error occurred while running the tool. Please try again. Error: {str(error)}"

ToolErrorFunction = Callable[[RunContextWrapper[Any], Exception], MaybeAwaitable[str]]

@overload
def function_tool(
    func: ToolFunction[...],
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    failure_error_function: ToolErrorFunction | None = None,
    strict_mode: bool = True,
) -> FunctionTool:
    """Overload for usage as @function_tool (no parentheses)."""
    ...

@overload
def function_tool(
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    failure_error_function: ToolErrorFunction | None = None,
    strict_mode: bool = True,
) -> Callable[[ToolFunction[...]], FunctionTool]:
    """Overload for usage as @function_tool(...)."""
    ...

def function_tool(
    func: ToolFunction[...] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    failure_error_function: ToolErrorFunction | None = default_tool_error_function,
    strict_mode: bool = True,
) -> FunctionTool | Callable[[ToolFunction[...]], FunctionTool]:
    """
    Decorator to create a FunctionTool from a function.
    Creates JSON schema from signature and uses docstring for descriptions.
    If function takes RunContextWrapper as first arg, it must match agent's context type.
    """

    def _create_function_tool(the_func: ToolFunction[...]) -> FunctionTool:
        schema = function_schema(
            func=the_func,
            name_override=name_override,
            description_override=description_override,
            docstring_style=docstring_style,
            use_docstring_info=use_docstring_info,
            strict_json_schema=strict_mode,
        )

        async def _on_invoke_tool_impl(ctx: RunContextWrapper[Any], input: str) -> Any:
            try:
                json_data: dict[str, Any] = json.loads(input) if input else {}
            except Exception as e:
                logger.debug(f"Invalid JSON input for tool {schema.name}: {input}")
                raise ModelError(
                    f"Invalid JSON input for tool {schema.name}: {input}"
                ) from e

            logger.debug(f"Invoking tool {schema.name} with input {input}")

            try:
                parsed = (
                    schema.params_pydantic_model(**json_data)
                    if json_data
                    else schema.params_pydantic_model()
                )
            except ValidationError as e:
                raise ModelError(f"Invalid JSON input for tool {schema.name}: {e}") from e

            args, kwargs_dict = schema.to_call_args(parsed)

            logger.debug(f"Tool call args: {args}, kwargs: {kwargs_dict}")

            try:
                result = the_func(ctx, *args, **kwargs_dict)
                if inspect.iscoroutine(result):
                    result = await result
                return str(result)
            except Exception as e:
                if failure_error_function:
                    error_msg = failure_error_function(ctx, e)
                    if inspect.iscoroutine(error_msg):
                        error_msg = await error_msg
                    return error_msg
                raise

        async def _on_invoke_tool(ctx: RunContextWrapper[Any], input: str) -> Any:
            try:
                return await _on_invoke_tool_impl(ctx, input)
            except Exception as e:
                logger.debug(f"Tool {schema.name} failed with error: {e}")
                raise

        return FunctionTool(
            name=schema.name,
            description=schema.description or "",
            params_json_schema=schema.params_json_schema,
            on_invoke_tool=_on_invoke_tool,
            strict_json_schema=strict_mode,
        )

    def decorator(real_func: ToolFunction[...]) -> FunctionTool:
        return _create_function_tool(real_func)

    if func is None:
        return decorator
    return decorator(func)
