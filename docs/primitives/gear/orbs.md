# orbs

<br>

## tl; dr

<br>

orbs serve as intelligent intermediaries that:

- facilitate dynamic task transfer between agents
- maintain context and state during delegation
- enable flexible agent-to-agent communication
- support complex multi-agent workflows

<br>

---

## contents

<br>

- [overview of the shield module](#overview-of-the-orbs-module)
    - [the `Orbs` class](#the-orbs-class)
- [tips and best practices](#tips-and-best-practices)
  - [customizing error messages](#customizing-error-messages)
  - [running tests](#running-tests)
- [available examples](#available-examples)

<br>

---

## overview of the orbs module

<br>

### the `Orbs` class

<br>

```python
@dataclass(frozen=True)
class OrbsInputData:
    """Data structure for input filtering during orbs operations."""

    input_history: str | tuple[InputItem, ...]
    pre_orbs_items: tuple[RunItem, ...]
    new_items: tuple[RunItem, ...]


# function that filters input data passed to the next agent
OrbsInputFilter = Callable[[OrbsInputData], OrbsInputData]


@dataclass
class Orbs(Generic[TContext]):
    """Represents delegation of a task from one agent to another."""

    on_invoke_orbs: Callable[[RunContextWrapper[TContext], str | None], Awaitable[Agent[TContext]]]
    name: str | None = None
    description: str | None = None
    input_json_schema: dict[str, Any] | None = None
    input_filter: OrbsInputFilter | None = None

    @classmethod
    def default_name(cls, agent: Agent[Any]) -> str:
        """Generate a default name for the orbs based on the agent name."""
        return transform_string_function_style(f"transfer_to_{agent.name}")

    @classmethod
    def default_description(cls, agent: Agent[Any]) -> str:
        """Generate a default description for the orbs based on the agent."""
        desc = agent.orbs_description or ""
        return f"Orbs to the {agent.name} agent to handle the request. {desc}"
```

<br>

where the parameters are:

<br>

| Parameter           | Type              | Description                    |
|---------------------|-------------------|--------------------------------|
| `name`              | `str`             | sword representing the orbs    |
| `description`       | `str`             | sword description              |
| `input_json_schema` | `dict`            | `JSON` schema for orbs input   |
| `input_filter`      | `OrbsInputFilter` | optional filter for orbs input |

<br>

and for filters:

<br>

| Parameter           | Type        | Description                        |
|---------------------|-------------|------------------------------------|
| `input_history`     | `str`       | input history before `Runner.run()`|
| `pre_orbs_items`    | `tuple`     | generated before the agent turn    |
| `new_items`         | `tuple`     | new items generated                |

<br>

a simple example of how orbs can be integrated to an agent is shown below:

<br>

```python
third_agent = Agent(
    name="Assistant Three",
    instructions="Replace one word in the sentence received from agent two with 'love' in a way that makes sense or is entertaining.",
    orbs_description="Replace one word in the input sentence with the word 'love'.",
)

second_agent = Agent(
    name="Agent Two",
    instructions="Create a sentence about the cypherpunk world with number of words exactly equal to the input number from agent one.",
    orbs=[orbs(third_agent, input_filter=orbs_message_filter)(transfer_to_third_agent)],
    orbs_description="Create sentences about the cypherpunk world.",
)

first_agent = Agent(
    name="Agent One",
    instructions="Generate a random between 3 and 15.",
    swords=[random_number],
)
```

<br>>

---

### the decorator `@orbs`

`@orbs` is implemented through a decorator factory utilizing the `@overload` pattern to provide better type hint and documentation for different ways the function can be called.

<br>

```python
def create_orbs_decorator(
    agent: Agent[T],
    input_filter: OrbsInputFilter | None = None,
) -> Callable:
    """Create an orbs decorator for the given agent."""

    def decorator(f: Any) -> Orbs[T]:
        input_json_schema: dict[str, Any] | None = None
        if hasattr(f, "__annotations__"):
            input_type = f.__annotations__.get("input_data", None)
            if input_type is not None:
                f.input_type = input_type
                try:
                    input_json_schema = input_type.model_json_schema()
                except (AttributeError, TypeError) as e:
                    raise set_error(UsageError, error_messages.ORBS_ERROR, error=str(e)) from e

        async def on_invoke(
            ctx: RunContextWrapper[T],
            input_json: str | None = None,
        ) -> Agent[T]:
            if hasattr(f, "input_type"):
                if input_json is None:
                    raise set_error(
                        UsageError,
                        error_messages.ORBS_ERROR,
                        error=f"{f.__name__}() missing 1 required positional argument: 'input_data'"
                    )
                try:
                    input_data = f.input_type.model_validate_json(input_json)
                    await _invoke_function(f, ctx, input_data)
                except Exception as e:
                    raise set_error(UsageError, error_messages.ORBS_ERROR, error=str(e)) from e
            else:
                await _invoke_function(f, ctx)
            return agent

        return Orbs(
            on_invoke_orbs=on_invoke,
            name=Orbs.default_name(agent),
            description=Orbs.default_description(agent),
            input_json_schema=input_json_schema,
            input_filter=input_filter,
        )

    return decorator


async def _invoke_function(
    func: Callable[..., Any],
    ctx: RunContextWrapper[Any],
    *args: Any,
) -> None:
    """Helper function to invoke a function, handling both sync and async cases."""
    if asyncio.iscoroutinefunction(func):
        await func(ctx, *args)
    else:
        func(ctx, *args)


# decorator for output orbs
orbs = create_orbs_decorator
```

<br>


----

## tips and best practices

<br>

### customizing error messages

<br>

in the code above, error handlers (and their messages) are stored inside `ORBS_ERROR_HANDLER`, which is defined in the top of the file with:

```python
ORBS_ERROR_HANDLER = create_error_handler(ERROR_MESSAGES.ORBS_ERROR.message)
```

<br>

`create_error_handler()` is a method defined in [src/util/exceptions.py](../../src/util/exceptions.py) and is not intended to be modified. however, the string `ERROR_MESSAGES.ORBS_ERROR.message` (which is imported from [src/util/constants.py](../../src/util/constants.py)) can be directly customized inside your [`.env`](../../../.env.example).

<br>

---

### running tests

<br>

unit tests for the `Orbs` module can be run with:

<br>

```shell
poetry run pytest tests/gear/test_orbs.py -v
```

<br>

---

## available examples

<br>

#### simple orbs by an agent

* [agent world traveler](../../examples/agents/world_traveler.py)


#### advanced orbs by an agent

* [agent friend with benefit](../../examples/agents/friend_with_benefits.py)
