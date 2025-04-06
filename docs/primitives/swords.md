# swords

<br>


swords let agents take actions such as: 

- running code
- fetching data 
- calling external APIs

<br>

## function swords

<br>

function swords are implemented in the class [gear/swords.py](../../src/gear/swords.py) and let you use one or more python function as the agent's "swords".

the name of the sword is the same name of the python function (python's inspect module extracts the function signature and griffe to parse docstrings and pydantic into schema creation):

```python
@dataclass(frozen=True)
class FunctionSword:
    name: str
    description: str
    params_json_schema: dict[str, Any]
    on_invoke_sword: Callable[[RunContextWrapper[Any], str], Awaitable[Any]]
    strict_json_schema: bool = True

@overload
def function_sword(
    func: SwordFunction[...],
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    failure_error_function: SwordErrorFunction | None = None,
    strict_mode: bool = True,
) -> FunctionSword:
    ...


@overload
def function_sword(
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    failure_error_function: SwordErrorFunction | None = None,
    strict_mode: bool = True,
) -> Callable[[SwordFunction[...]], FunctionSword]:
    ...


def function_sword(
    func: SwordFunction[...] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    failure_error_function: SwordErrorFunction | None = default_sword_error_function,
    strict_mode: bool = True,
) -> FunctionSword | Callable[[SwordFunction[...]], FunctionSword]:
    def _create_function_sword(the_func: SwordFunction[...]) -> FunctionSword:
        schema = function_schema(
            func=the_func,
            name_override=name_override,
            description_override=description_override,
            docstring_style=docstring_style,
            use_docstring_info=use_docstring_info,
            strict_json_schema=strict_mode,
        )

        async def _on_invoke_sword_impl(ctx: RunContextWrapper[Any], input: str) -> Any:
            try:
                json_data: dict[str, Any] = json.loads(input) if input else {}
                parsed = (
                    schema.params_pydantic_model(**json_data)
                    if json_data
                    else schema.params_pydantic_model()
                )
                args, kwargs_dict = schema.to_call_args(parsed)
                logger.debug(f"Sword call args: {args}, kwargs: {kwargs_dict}")

                result = the_func(ctx, *args, **kwargs_dict)
                if inspect.iscoroutine(result):
                    result = await result
                return str(result)
            except json.JSONDecodeError as e:
                logger.debug(f"Invalid JSON input for sword {schema.name}: {input}")
                raise ModelError(f"Invalid JSON input for sword {schema.name}: {input}") from e
            except ValidationError as e:
                raise ModelError(f"Invalid JSON input for sword {schema.name}: {e}") from e
            except Exception as e:
                if failure_error_function:
                    error_msg = failure_error_function(ctx, e)
                    if inspect.iscoroutine(error_msg):
                        error_msg = await error_msg
                    return error_msg
                raise GenericError(e) from e

        async def _on_invoke_sword(ctx: RunContextWrapper[Any], input: str) -> Any:
            try:
                return await _on_invoke_sword_impl(ctx, input)
            except Exception as e:
                logger.debug(f"Sword {schema.name} failed with error: {e}")
                raise GenericError(e) from e

        return FunctionSword(
            name=schema.name,
            description=schema.description or "",
            params_json_schema=schema.params_json_schema,
            on_invoke_sword=_on_invoke_sword,
            strict_json_schema=strict_mode,
        )

    def decorator(real_func: SwordFunction[...]) -> FunctionSword:
        return _create_function_sword(real_func)

    if func is None:
        return decorator
    return decorator(func)
```

<br>

they can be used as a decorator:

```python
@function_sword
def random_number() -> int:
    return random.randint(3, 15)
```