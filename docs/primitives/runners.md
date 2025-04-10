# `Runner`

<br>

## tl; dr

<br>

the `Runner` class, located at [`src/runners/run.py`](../../src/runners/run.py), is the core orchestrator for executing agents in the drusilla system.

it provides both synchronous and asynchronous interfaces for running agents with various configurations and capabilities.

<br>

key features:

- asynchronous execution (`run`), returning a `RunResult`
- synchronous execution (`run_sync`) (a sync method that runs `.run()`)
- streaming execution (`run_streamed`) for real-time agent responses, returning a `RunResultStreaming`

<br>

---

## contents

<br>

- [overview of the `Runner` class](#overview-of-the-runner-class)
  - [configuring the agent run](#configuring-the-agent-run) 
  - [run results](#run-results)
  - [run streaming](#run-streaming)
  - [usability examples](#usability-examples)
- [tips and best practices](#tips-and-best-practices)
  - [running tests](#running-tests)

<br>

---

## overview of the `Runner` class

<br>

the `Runner`'s `run()` method receives an `Agent` object and an `input`. the input can be either a string (such as a user message) or a list of input items.

the runner then runs a loop calling the LLM to produce an output, with the following possible outcomes:

- if a `final_output` is returned, the loop ends with the result
- if the LLM calls for orbs, the current agent and input are updated and the loop is re-run 
- if the LLM calls for swords, they run, append the results, and re-run the loop
- if `max_turns` (set in `.env`) is exceeded, a `MaxTurnsError` exception is raised

<br>

the `Runner` class structure is shown below:

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
        if input is None:
            raise RunnerError(err.RUNNER_ERROR.format(error="Invalid input: input cannot be None"))
        if max_turns <= 0:
            raise RunnerError(err.RUNNER_ERROR.format(error="Max turns must be positive"))

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
        max_turns: int = config.MAX_TURNS,
        charms: RunCharms[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResult:
        try:
            charms, run_config, context_wrapper, input, _ = await cls._initialize_run(
                starting_agent, input, context, max_turns, charms, run_config
            )

            current_turn = 0
            original_input = input if isinstance(input, str) else input.copy()
            generated_items = []
            model_responses = []
            input_shield_results = []
            current_agent = starting_agent
            should_run_agent_start_charms = True

            while True:
                current_turn += 1
                if current_turn > run_config.max_turns:
                    raise MaxTurnsError(
                        config.RUNNER_ERROR.format(
                            error=f"Max turns ({max_turns}) exceeded"
                        )
                    )

                try:
                    turn_result = await cls._run_turn(
                        current_turn,
                        current_agent,
                        original_input,
                        generated_items,
                        charms,
                        context_wrapper,
                        run_config,
                        should_run_agent_start_charms,
                        input,
                    )
                except Exception as e:
                    error_message = err.AGENT_EXEC_ERROR.format(error=str(e))
                    raise AgentError(error_message) from e

                should_run_agent_start_charms = False
                model_responses.append(turn_result.model_response)
                original_input = turn_result.original_input
                generated_items = turn_result.generated_items

                if isinstance(turn_result.next_step, NextStepFinalOutput):
                    return await cls._handle_final_output(
                        current_agent,
                        turn_result,
                        original_input,
                        generated_items,
                        model_responses,
                        input_shield_results,
                        context_wrapper,
                        run_config,
                    )
                elif isinstance(turn_result.next_step, NextStepOrbs):
                    current_agent = cast(Agent[TContext], turn_result.next_step.new_agent)
                    should_run_agent_start_charms = True
                elif isinstance(turn_result.next_step, NextStepRunAgain):
                    continue
                else:
                    raise RunnerError(
                        err.RUNNER_ERROR.format(
                            error=f"Unknown next step type: {type(turn_result.next_step)}"
                        )
                    )
        except Exception as e:
            raise RunnerError(err.RUNNER_ERROR.format(error=str(e))) from e

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[InputItem],
        *,
        context: TContext | None = None,
        max_turns: int = config.MAX_TURNS,
        charms: RunCharms[TContext] | None = None,
        run_config: RunConfig | None = None,
        timeout: float | None = None,
    ) -> RunResult:
        """Synchronous wrapper for run()."""
        return asyncio.get_event_loop().run_until_complete(
            asyncio.wait_for(
                cls.run(
                    starting_agent,
                    input,
                    context=context,
                    max_turns=max_turns,
                    charms=charms,
                    run_config=run_config,
                ),
                timeout=timeout,
            )
        )

    @classmethod
    async def run_streamed(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[InputItem],
        context: TContext | None = None,
        max_turns: int = config.MAX_TURNS,
        charms: RunCharms[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResultStreaming:
        charms, run_config, context_wrapper, input, __output_schema = await cls._initialize_run(
            starting_agent, input, context, max_turns, charms, run_config
        )

        streamed_result = cls._create_streamed_result(
            input=input,
            starting_agent=starting_agent,
            max_turns=run_config.max_turns,
        )

        streamed_result._run_impl_task = asyncio.create_task(
            cls._run_streamed_impl(
                starting_input=input,
                streamed_result=streamed_result,
                starting_agent=starting_agent,
                max_turns=max_turns,
                charms=charms,
                context_wrapper=context_wrapper,
                run_config=run_config,
            )
        )
        return streamed_result
```

<br>

---

### configuring the agent run

<br>

the `run_config` parameter gets a `RunConfig` object (located at [src/runners/config.py](../../src/runners/config.py)) for global settings configuration of the agent run:

<br>

```python
@dataclass
class RunConfig:
    """Configuration for running an agent."""

    max_turns: int = config.MAX_TURNS
    max_tokens: int = config.MAX_TOKENS
    temperature: float = config.TEMPERATURE
    top_p: float = config.TOP_P
    sword_choice: str = config.SWORD_CHOICE
    parallel_sword_calls: bool = config.PARALLEL_SWORD_CALLS
```

<br>

---

### run results

<br>

the `run()` method returns a `RunResult` object, which contains:

<br>

```python
@dataclass
class RunResult:
    """Result of running an agent."""

    final_output: Any
    model_responses: list[ModelResponse]
    input_shield_results: list[InputShieldResult]
    context: Any
    run_config: RunConfig
```

<br>

---

### run streaming

<br>

the `run_streamed()` method returns a `RunResultStreaming` object, which contains:

<br>

```python
@dataclass
class RunResultStreaming:
    """Result of running an agent with streaming."""

    input: str | list[InputItem]
    starting_agent: Agent[Any]
    max_turns: int
    _run_impl_task: asyncio.Task[None] | None = None
```

<br>

---

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

---

## tips and best practices

<br>

### customizing error messages

<br>

in the code above, error handlers (and their messages) are held inside `RUNNER_ERROR`.

`create_error_handler()` is a method defined inside [util/_exceptions.py](../../src/util/_exceptions.py) and is not intended to be modified. however, the string `err.RUNNER_ERROR` (which is imported from [util/_constants.py](../../src/util/_constants.py)) can be directly customized inside your [`.env`](../../.env.example).

<br>

---

### running tests

<br>

unit tests for `Runner` can be run with:

<br>

```shell
poetry run pytest tests/agents/test_runner.py -v
```
