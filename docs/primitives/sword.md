# swords

<br>

## tl; dr

<br>

swords are specialized tools that wrap python functions with enhanced capabilities, letting agents take actions such as:

- running code
- fetching data
- calling external APIs

<br>

the sword primitive is defined in the [src/gear/sword.py](../../src/gear/swords.py) module. swords can be defined by:

- functions and decorators (function swords), or
- an agent (agent swords)

<br>

---

## contents
- [overview of the sword module](#overview-of-the-sword-module)
    - [the main class `Sword`](#the-main-class-sword)
    - [creating swords with python functions](#creating-swords-with-python-functions)
    - [creating a custom `Sword` object](#creating-a-custom-sword-object)
    - [creating an agent as a sword](#creating-an-agent-as-a-sword)
- [tests for `sword.py`](#tests-for-swordpy)
- [tips and best practices](#tips-and-best-practices)
  - [customizing error messages](#customizing-error-messages)
- [advanced examples](#advanced-examples)


<br>

---

## overview of the sword module

<br>

### the main class `Sword`

<br>

the driving classes of this primitive are the dataclasses `Sword` and `SwordResult`, defined as:

<br>

```python
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
```

<br>

where `RunItem` is a union type that represents different types of items that can be generated during an agent's run.

<br>

---

### creating swords with python functions

<br>

the easiest way to create a sword is by using any python function. they can become a sword by attaching the decorator `@function_sword':

1. the name of the sword will be the name of the python function
2. the description is the docstring of the function
3. the schema for the function inputs is created from the function's arguments
4. the function signature is extracted by the python's inspect module, along with [pydantic](https://docs.pydantic.dev/latest/) for schema creation

<br>

`@function_sword` is implemented through the `@overload` pattern, to provide better type hint and documentation for different ways the function can be called:

<br>

```python
@overload
def function_sword(
    func: SwordFunction[...],
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    failure_error_function: SwordErrorFunction | None = SWORD_ERROR_HANDLER,
    strict_mode: bool = True,
) -> Sword: ...


@overload
def function_sword(
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    failure_error_function: SwordErrorFunction | None = SWORD_ERROR_HANDLER,
    strict_mode: bool = True,
) -> Callable[[SwordFunction[...]], Sword]: ...


def function_sword(
    func: SwordFunction[...] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    failure_error_function: SwordErrorFunction | None = SWORD_ERROR_HANDLER,
    strict_mode: bool = True,
) -> Sword | Callable[[SwordFunction[...]], Sword]:

    def create_sword(f: SwordFunction[...]) -> Sword:
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
                    msg = failure_error_function(ctx, e)
                    return await msg if inspect.iscoroutine(msg) else msg
                raise

        return Sword(
            name=schema.name,
            description=schema.description,
            params_json_schema=schema.params_json_schema,
            on_invoke_sword=on_invoke,
            strict_json_schema=strict_mode,
            failure_error_function=failure_error_function,
        )

    return create_sword if func is None else create_sword(func)
```

<br>

the schemas above are defined by the dataclass `FuncSchema` and the `function_schema()` method:

<br>

```python
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
                # Parameters with default values are treated as keyword arguments
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

    # Create dynamic model and schema
    dynamic_model = create_model(f"{func_name}_args", __base__=BaseModel, **fields)
    json_schema = dynamic_model.model_json_schema()
    if strict_json_schema:
        json_schema = ensure_strict_json_schema(json_schema)

    # Create invocation handler
    on_invoke_sword = _create_invocation_handler(
        func,
        dynamic_model,
        sig,
        takes_context,
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
```

<br>

the invocation within these function schemas is defined by two private methods:

<br>

```python
def _create_invocation_handler(
    func: Callable[..., Any],
    dynamic_model: type[BaseModel],
    sig: inspect.Signature,
    takes_context: bool,
) -> Callable[[RunContextWrapper[Any], str], Awaitable[Any]]:
    """Create the sword invocation handler."""

    async def on_invoke_sword(ctx: RunContextWrapper[Any], input: str) -> Any:
        data = dynamic_model.model_validate_json(input)
        args, kwargs = FuncSchema(
            name=func.__name__,
            description=None,
            params_pydantic_model=dynamic_model,
            params_json_schema=dynamic_model.model_json_schema(),
            signature=sig,
            on_invoke_sword=lambda _, __: None,
            takes_context=takes_context,
            strict_json_schema=True,
        ).to_call_args(data)

        if takes_context:
            args.insert(0, ctx)

        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

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
                raise UsageError(ERROR_MESSAGES.RUN_CONTEXT_ERROR.message.format(error=(sig.name)))
        filtered_params.append((name, param))

    return takes_context, filtered_params
```

<br>

finally, the documentation is parsed using the following methods and dataclass:

<br>

```python
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
    """Create Pydantic fields from function parameters."""
    fields: dict[str, tuple[Any, Field]] = {}

    for name, param in params:
        ann = type_hints.get(name, param.annotation)
        default = param.default
        field_description = param_descs.get(name)

        if ann == inspect._empty:
            ann = Any

        if param.kind == param.VAR_POSITIONAL:
            # Handle *args - always use list[Any]
            ann = list[Any]  # type: ignore
            field = Field(default_factory=list, description=field_description)
        elif param.kind == param.VAR_KEYWORD:
            # Handle **kwargs - always use dict[str, Any]
            ann = dict[str, Any]  # type: ignore
            field = Field(default_factory=dict, description=field_description)
        else:
            field = Field(
                ... if default == inspect._empty else default,
                description=field_description,
            )

        fields[name] = (ann, field)

    return fields
```

<br>

----

### creating a custom `Sword` object

<br>

alternatively, we might want to create a custom sword instead of using an existing python function. in this case, you can use `Sword` and provide:

- a name for the sword
- a description
- `params_json_schema`, with the `JSON` schema for the arguments
`on_invoke_sword` (the async function defined inside `_create_invocation_handler` that receives the context and the arguments as a `JSON` string, and returns the sword output as `str`)

<br>

note that, again, the function signature is parsed to extract the schema for the sword, and the docstring is parsed to extract descriptions for the sword and for individual arguments (defined by `FuncSchema` and `function_schema`).

this pseudo code illustrate this approach (where `model_validate_json()` and `model_validate_json()` are pydantic methods):

<br>

```python
class SwordArgs():
    name: str
    color: str

sword = Sword(
    name="user",
    description="parse information",
    params_json_schema=SwordArgs.model_json_schema(),
    on_invoke_tool=run_sword_func,
)

async def run_sword_func(_ctx_: RunContextWrapper[Any], args: str) -> str:
    data = SwordArgs.model_validate_json(args)
    return do_some_work(data=f"{data.name} likes {data.color}")
```

<br>

---

### creating an agent as a sword

<br>

a [charms](charm.md) (the agent's workflow) may use an agent to orchestrate a network of specialized agents (instead of handing off control). this can be done by modeling agents as swords.

a simple example is given on our [agent world traveler](../../examples/agents/world_traveler.py):

<br>

```python
def create_agents() -> Agent:
    return Agent(
        name="Agent World Traveler",
        instructions=(
            "You are a cool special robot who coordinates translation requests."
            "Use appropriate translation swords based on requested languages."
        ),
        swords=[
            Agent(
                name=f"{lang_code.upper()} Translator",
                instructions=f"Translate English text to {lang_code.upper()}",
                orbs_description=f"English to {lang_code.upper()} translator",
            ).as_sword(
                sword_name=f"translate_to_{lang_code.lower()}",
                sword_description=f"Translate text to {lang_code.upper()}",
            )
            for lang_code in SUPPORTED_LANGUAGES
        ],
    )
```

<br>

### tests for `sword.py`

<br>

unit tests for the sword primitive can be found inside [test/gear](../../tests/gear/test_sword.py):

<br>

```python
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

        def test_func(a: int, b: str, c: list[int] | None = None, d: dict[str, Any] | None = None) -> None:
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
        @function_sword(
            name_override="custom_sword",
            description_override="Custom description"
        )
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
```

<br>

----

## tips and best practices

<br>

### customizing error messages

<br>

in the code above, error handlers (and their messages) are held inside `SWORD_ERROR_HANDLER`, which is defined in the top of the file with:

<br>

```python
SWORD_ERROR_HANDLER = create_error_handler(ERROR_MESSAGES.SWORD_ERROR.message)
```

<br>

`create_error_handler()` is a method defined inside [util/_exceptions.py](../../src/util/_exceptions.py) and is not intended to be modified. however, the string `ERROR_MESSAGES.SWORD_ERROR.message` (which is imported from [util/_constants.py](../../src/util/_constants.py)) can be directly customized inside your [`.env`](../../.env.example).


<br>

---

## advanced examples
