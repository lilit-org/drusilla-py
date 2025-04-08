# `Runner`

<br>

## tl; dr

<br>

the `Runner` class, located at [`src/runners/run.py`](../../src/runners/run.py), is the core orchestrator for executing agents in the drusilla system.

it provides both synchronous and asynchronous interfaces for running agents with various configurations and capabilities.

<br>

key features:

- supports both synchronous (`run_sync`) and asynchronous (`run`) execution
- provides streaming capabilities (`run_streamed`) for real-time agent responses

<br>

---

## contents

<br>

- [overview of the `Runner` class](#overview-of-the-runner-class)
  - [usability examples](#usability-examples)
- [tips and best practices](#tips-and-best-practices)
  - [running tests](#running-tests)

<br>

---

## overview of the `Runner` class

<br>

```python
class Runner:
    @classmethod
    async def _initialize_run(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[InputItem] | None,
        context: TContext | None,
        max_turns: int,
        charms: RunCharms[TContext] | None,
        run_config: RunConfig | None,
    ) -> tuple[
        RunCharms[TContext],
        RunConfig,
        RunContextWrapper[TContext],
        str | list[InputItem],
        AgentOutputSchema | None,
    ]:

        charms = charms or RunCharms[Any]()
        run_config = run_config or RunConfig()
        config_dict = {k: v for k, v in run_config.__dict__.items() if k != "max_turns"}
        _output_schema = cls._get_output_schema(starting_agent)
        return (
            charms,
            RunConfig(**config_dict, max_turns=max_turns),
            RunContextWrapper(context=context),
            input,
            _output_schema,
        )

    @classmethod
    async def run(
            cls,
            starting_agent: Agent[TContext],
            input: str | list[InputItem],
            *,
            context: TContext | None = None,
            max_turns: int = MAX_TURNS,
            charms: RunCharms[TContext] | None = None,
            run_config: RunConfig | None = None,
    ) -> RunResult:
            ...

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[InputItem],
        *,
        context: TContext | None = None,
        max_turns: int = MAX_TURNS,
        charms: RunCharms[TContext] | None = None,
        run_config: RunConfig | None = None,
        timeout: float | None = None,
    ) -> RunResult:
            ...

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[InputItem],
        *,
        context: TContext | None = None,
        max_turns: int = MAX_TURNS,
        charms: RunCharms[TContext] | None = None,
        run_config: RunConfig | None = None,
        timeout: float | None = None,
    ) -> RunResult:
            ...

    @classmethod
    async def run_streamed(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[InputItem],
        context: TContext | None = None,
        max_turns: int = MAX_TURNS,
        charms: RunCharms[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResultStreaming:
            ...
```

<br>

----

### usability examples

<br>

running it synchronously:

<br>

```python
result = Runner.run_sync(agent, msg)
```

<br>

running it asynchronously:

<br>

```python
result = await Runner.run_streamed(
            agent,
            f"Tell me exactly {num_jokes} cypherpunk jokes!",
        )
await _handle_stream_events(result, num_jokes)
```

<br>

----

## tips and best practices

<br>

### customizing error messages

<br>

in the code above, error handlers (and their messages) are held inside `RUNNER_ERROR`.

`create_error_handler()` is a method defined inside [util/_exceptions.py](../../src/util/_exceptions.py) and is not intended to be modified. however, the string `ERROR_MESSAGES.SWORD_ERROR.message` (which is imported from [util/_constants.py](../../src/util/_constants.py)) can be directly customized inside your [`.env`](../../.env.example).

<br>

---

### running tests

<br>

unit tests for `Runner` can be run with:

<br>

```shell
poetry run pytest tests/agents/test_runner.py -v
```
