# shields

<br>

## tl; dr

<br>

shields are used to validate the agent's inputs and outputs, running in parallel and ensuring data integrity and safety throughout the agent execution pipeline.

there are two types of shields:
- `InputShield`: validates and sanitizes agent input before execution
- `OutputShield`: validates and formats agent output after execution


<br>

---

## contents

<br>

- [overview of the shield module](#overview-of-the-shield-module)
    - [the base class for `Shield`](#the-base-class-for-shield)
    - [the `InputShield` dataclass](#the-inputshield-dataclass)
    - [the `OutputShield` dataclass](#the-outputshield-dataclass)
- [tips and best practices](#tips-and-best-practices)
  - [customizing error messages](#customizing-error-messages)
  - [running tests](#running-tests)
- [advanced examples](#advanced-examples)

<br>

---

## overview of the shield module

<br>

### the base class for `Shield`

<br>

the base dataclasses for both `InputShield` and `OutputShield` are defined as follows:

<br>

```python
@dataclass(frozen=True)
class ShieldResult:
    """Result of a shield validation operation."""

    success: bool
    message: str | None = None
    data: Any | None = None
    tripwire_triggered: bool = False
    result: Any | None = None


@dataclass(frozen=True)
class ShieldResultWrapper(Generic[T]):
    """Wrapper for shield results with additional context."""

    shield: Any
    agent: Agent[Any]
    data: T
    output: ShieldResult


class BaseShield(Generic[T, TContext]):
    """Base class for shields with common functionality."""

    def __init__(
        self,
        shield_function: Callable[
            [RunContextWrapper[TContext], Agent[Any], T],
            MaybeAwaitable[ShieldResult],
        ],
        name: str | None = None,
    ):
        self.shield_function = shield_function
        self.name = name

    async def _run_common(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[Any],
        data: T,
    ) -> ShieldResult:
        if not callable(self.shield_function):
            error_msg = ERROR_MESSAGES.SHIELD_FUNCTION_ERROR.message.format(
                error=self.shield_function
            )
            raise UsageError(error_msg)

        output = self.shield_function(context, agent, data)
        if inspect.isawaitable(output):
            output = await output

        return output
```

<br>

a shield decorator can then be then created by this method:

<br>

```python
def create_shield_decorator(
    shield_class: type[BaseShield[TContext_co, Any]],
    sync_func_type: type,
    async_func_type: type,
):
    return create_decorator_factory(shield_class, sync_func_type, async_func_type)
```

<br>

----

### the `InputShield` dataclass

<br>

```python
@dataclass(frozen=True)
class InputShieldResult:
    """Result from an input shield function."""

    tripwire_triggered: bool
    shield: Any
    agent: Agent[Any]
    input: str | list[InputItem]
    output: ShieldResult
    result: Any | None = None


class InputShield(BaseShield[str | list[InputItem], TContext]):
    """Shield that validates agent input before execution."""

    async def run(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[Any],
        input: str | list[InputItem],
    ) -> InputShieldResult:
        result = await self._run_common(context, agent, input)
        return InputShieldResult(
            tripwire_triggered=not result.success or result.tripwire_triggered,
            result=result.message if not result.success else result.data,
            shield=self,
            agent=agent,
            input=input,
            output=result,
        )
```

<br>

we can put everything together and create the decorator `@input_shield`:

<br>

```python
# typeclass for input shield
InputShieldFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", str | list[ResponseInputItemParam]],
    ShieldResult,
]
InputShieldFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", str | list[ResponseInputItemParam]],
    MaybeAwaitable[ShieldResult],
]

# decorator for input shield
input_shield = create_shield_decorator(
    InputShield,
    InputShieldFuncSync,
    InputShieldFuncAsync,
)
```

<br>

----

### the `OutputShield` dataclass

<br>

```python
@dataclass(frozen=True)
class OutputShieldResult:
    """Result from an output shield function."""

    tripwire_triggered: bool
    shield: Any
    agent: Agent[Any]
    agent_output: Any
    output: ShieldResult
    result: Any | None = None


class OutputShield(BaseShield[Any, TContext]):
    """Shield that validates agent output after execution."""

    async def run(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[Any],
        agent_output: Any,
    ) -> OutputShieldResult:
        result = await self._run_common(context, agent, agent_output)
        return OutputShieldResult(
            tripwire_triggered=not result.success or result.tripwire_triggered,
            result=result.message if not result.success else result.data,
            shield=self,
            agent=agent,
            agent_output=agent_output,
            output=result,
        )
```

<br>

similarly, we create the `@output_shield` decorator with:

<br>

```python
# typeclass for input shield
InputShieldFuncSync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", str | list[ResponseInputItemParam]],
    ShieldResult,
]
InputShieldFuncAsync = Callable[
    [RunContextWrapper[TContext_co], "Agent[Any]", str | list[ResponseInputItemParam]],
    MaybeAwaitable[ShieldResult],
]

# decorator for input shield
input_shield = create_shield_decorator(
    InputShield,
    InputShieldFuncSync,
    InputShieldFuncAsync,
)
```


<br>

----

## tips and best practices

<br>

### customizing error messages

<br>

in the code above, error handlers (and their messages) are stored inside `SHIELD_ERROR_HANDLER`, which is defined in the top of the file with:

```python
SHIELD_ERROR_HANDLER = create_error_handler(ERROR_MESSAGES.SHIELD_ERROR.message)
```

<br>

`create_error_handler()` is a method defined in [util/_exceptions.py](../../src/util/_exceptions.py) and is not intended to be modified. however, the string `ERROR_MESSAGES.SHIELD_ERROR.message` (which is imported from [util/_constants.py](../../src/util/_constants.py)) can be directly customized inside your [`.env`](../../.env.example).

<br>

### running tests

<br>

unit tests for the `Shield` module can be run with:

<br>

```shell
poetry run pytest tests/gear/test_shield.py -v
```

<br>

---

## advanced examples
