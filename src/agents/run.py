from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from ..models.interface import Model
from ..models.provider import ModelProvider
from ..models.settings import ModelSettings
from ..util._constants import DEFAULT_MAX_TURNS
from ..util._env import get_env_var
from ..util._exceptions import (
    AgentError,
    GenericError,
    InputGuardrailError,
    MaxTurnsError,
    ModelError,
    OutputGuardrailError,
)
from ..util._guardrail import (
    InputGuardrail,
    InputGuardrailResult,
    OutputGuardrail,
    OutputGuardrailResult,
)
from ..util._handoffs import Handoff, HandoffInputFilter, handoff
from ..util._items import ItemHelpers, ModelResponse, RunItem, TResponseInputItem
from ..util._lifecycle import RunHooks
from ..util._result import RunResult, RunResultStreaming
from ..util._run_context import RunContextWrapper, TContext
from ..util._stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent
from ..util._types import ResponseCompletedEvent, Usage
from .agent import Agent
from .output import AgentOutputSchema
from .run_impl import (
    NextStepFinalOutput,
    NextStepHandoff,
    NextStepRunAgain,
    QueueCompleteSentinel,
    RunImpl,
    SingleStepResult,
    noop_coroutine,
)

########################################################
#             Constants                                #
########################################################

MAX_TURNS = get_env_var("MAX_TURNS", DEFAULT_MAX_TURNS)


########################################################
#               Data classes                            #
########################################################

@dataclass(frozen=True)
class RunConfig:
    """Settings for agent run."""

    model: str | Model | None = None
    model_provider: ModelProvider = field(default_factory=ModelProvider)
    model_settings: ModelSettings | None = None
    handoff_input_filter: HandoffInputFilter | None = None
    input_guardrails: list[InputGuardrail[Any]] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail[Any]] = field(default_factory=list)
    max_turns: int = MAX_TURNS


########################################################
#               Main Class                            #
########################################################

class Runner:
    @classmethod
    async def _initialize_run(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        context: TContext | None,
        max_turns: int,
        hooks: RunHooks[TContext] | None,
        run_config: RunConfig | None,
    ) -> tuple[RunHooks[TContext], RunConfig, RunContextWrapper[TContext], str | list[TResponseInputItem], AgentOutputSchema | None]:

        hooks = hooks or RunHooks[Any]()
        run_config = run_config or RunConfig()
        run_config = RunConfig(
            model=run_config.model,
            model_provider=run_config.model_provider,
            model_settings=run_config.model_settings,
            handoff_input_filter=run_config.handoff_input_filter,
            input_guardrails=run_config.input_guardrails,
            output_guardrails=run_config.output_guardrails,
            max_turns=max_turns
        )
        context_wrapper = RunContextWrapper(context=context)
        output_schema = cls._get_output_schema(starting_agent)
        return hooks, run_config, context_wrapper, input, output_schema

    @classmethod
    async def run(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResult:

        hooks, run_config, context_wrapper, input, _ = await cls._initialize_run(
            starting_agent, input, context, max_turns, hooks, run_config
        )

        current_turn = 0
        original_input = input if isinstance(input, str) else input.copy()
        generated_items: list[RunItem] = []
        model_responses: list[ModelResponse] = []
        input_guardrail_results: list[InputGuardrailResult] = []
        current_agent = starting_agent
        should_run_agent_start_hooks = True

        try:
            while True:
                current_turn += 1
                if current_turn > run_config.max_turns:
                    raise MaxTurnsError(f"Max turns ({run_config.max_turns}) exceeded")

                turn_result = await cls._run_turn(
                    current_turn,
                    current_agent,
                    original_input,
                    generated_items,
                    hooks,
                    context_wrapper,
                    run_config,
                    should_run_agent_start_hooks,
                    input,
                )

                should_run_agent_start_hooks = False
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
                        input_guardrail_results,
                        context_wrapper,
                        run_config,
                    )
                elif isinstance(turn_result.next_step, NextStepHandoff):
                    current_agent = cast(Agent[TContext], turn_result.next_step.new_agent)
                    should_run_agent_start_hooks = True
                elif isinstance(turn_result.next_step, NextStepRunAgain):
                    continue
                else:
                    raise AgentError(f"Unknown next step type: {type(turn_result.next_step)}")
        except Exception as e:
            raise GenericError(e)

    @classmethod
    def _handle_max_turns_exceeded(cls, max_turns: int) -> None:
        raise MaxTurnsError(f"Max turns ({max_turns}) exceeded")

    @classmethod
    async def _run_turn(
        cls,
        current_turn: int,
        current_agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        generated_items: list[RunItem],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_hooks: bool,
        input: str | list[TResponseInputItem],
    ) -> SingleStepResult:

        if current_turn == 1:
            _, turn_result = await asyncio.gather(
                cls._run_input_guardrails(
                    current_agent,
                    current_agent.input_guardrails + run_config.input_guardrails,
                    deepcopy(input),
                    context_wrapper,
                ),
                cls._run_single_turn(
                    agent=current_agent,
                    original_input=original_input,
                    generated_items=generated_items,
                    hooks=hooks,
                    context_wrapper=context_wrapper,
                    run_config=run_config,
                    should_run_agent_start_hooks=should_run_agent_start_hooks,
                ),
            )
        else:
            turn_result = await cls._run_single_turn(
                agent=current_agent,
                original_input=original_input,
                generated_items=generated_items,
                hooks=hooks,
                context_wrapper=context_wrapper,
                run_config=run_config,
                should_run_agent_start_hooks=should_run_agent_start_hooks,
            )
        return turn_result

    @classmethod
    async def _handle_final_output(
        cls,
        current_agent: Agent[TContext],
        turn_result: SingleStepResult,
        original_input: str | list[TResponseInputItem],
        generated_items: list[RunItem],
        model_responses: list[ModelResponse],
        input_guardrail_results: list[InputGuardrailResult],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> RunResult:

        output_guardrail_results = await cls._run_output_guardrails(
            current_agent.output_guardrails + run_config.output_guardrails,
            current_agent,
            turn_result.next_step.output,
            context_wrapper,
        )
        return RunResult(
            input=original_input,
            new_items=generated_items,
            raw_responses=model_responses,
            final_output=turn_result.next_step.output,
            _last_agent=current_agent,
            input_guardrail_results=input_guardrail_results,
            output_guardrail_results=output_guardrail_results,
        )

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResult:
        """Synchronous wrapper for run()."""
        return asyncio.get_event_loop().run_until_complete(
            cls.run(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
            )
        )

    @classmethod
    def run_streamed(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        context: TContext | None = None,
        max_turns: int = MAX_TURNS,
        hooks: RunHooks[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResultStreaming:
        """Run agent with streaming events."""
        hooks, run_config, context_wrapper, input, output_schema = asyncio.get_event_loop().run_until_complete(
            cls._initialize_run(starting_agent, input, context, max_turns, hooks, run_config)
        )

        streamed_result = cls._create_streamed_result(
            input=input,
            starting_agent=starting_agent,
            max_turns=run_config.max_turns,
            output_schema=output_schema,
        )

        streamed_result._run_impl_task = asyncio.create_task(
            cls._run_streamed_impl(
                starting_input=input,
                streamed_result=streamed_result,
                starting_agent=starting_agent,
                max_turns=max_turns,
                hooks=hooks,
                context_wrapper=context_wrapper,
                run_config=run_config,
            )
        )
        return streamed_result

    @classmethod
    def _create_streamed_result(
        cls,
        input: str | list[TResponseInputItem],
        starting_agent: Agent[TContext],
        max_turns: int,
        output_schema: AgentOutputSchema | None,
    ) -> RunResultStreaming:
        """Create a new streaming result with optimized initialization."""
        return RunResultStreaming(
            input=input if isinstance(input, str) else input.copy(),
            new_items=[],
            current_agent=starting_agent,
            raw_responses=[],
            final_output=None,
            is_complete=False,
            current_turn=0,
            max_turns=max_turns,
            input_guardrail_results=[],
            output_guardrail_results=[],
            _current_agent_output_schema=output_schema,
        )

    @classmethod
    async def _run_streamed_impl(
        cls,
        starting_input: str | list[TResponseInputItem],
        streamed_result: RunResultStreaming,
        starting_agent: Agent[TContext],
        max_turns: int,
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> None:

        current_agent = starting_agent
        current_turn = 0
        should_run_agent_start_hooks = True
        streamed_result._event_queue.put_nowait(AgentUpdatedStreamEvent(new_agent=current_agent))

        try:
            while not streamed_result.is_complete:
                current_turn += 1
                streamed_result.current_turn = current_turn

                if current_turn > max_turns:
                    cls._handle_max_turns_exceeded(max_turns)
                    streamed_result._event_queue.put_nowait(QueueCompleteSentinel())
                    break

                if current_turn == 1:
                    streamed_result._input_guardrails_task = asyncio.create_task(
                        cls._run_input_guardrails_with_queue(
                            starting_agent,
                            starting_agent.input_guardrails + run_config.input_guardrails,
                            deepcopy(ItemHelpers.input_to_new_input_list(starting_input)),
                            context_wrapper,
                            streamed_result,
                        )
                    )

                try:
                    turn_result = await cls._run_single_turn_streamed(
                        streamed_result,
                        current_agent,
                        hooks,
                        context_wrapper,
                        run_config,
                        should_run_agent_start_hooks,
                    )
                    should_run_agent_start_hooks = False

                    cls._update_streamed_result(streamed_result, turn_result)

                    if isinstance(turn_result.next_step, NextStepHandoff):
                        current_agent = turn_result.next_step.new_agent
                        should_run_agent_start_hooks = True
                        streamed_result._event_queue.put_nowait(
                            AgentUpdatedStreamEvent(new_agent=current_agent)
                        )
                    elif isinstance(turn_result.next_step, NextStepFinalOutput):
                        await cls._handle_streamed_final_output(
                            current_agent,
                            turn_result,
                            streamed_result,
                            context_wrapper,
                            run_config,
                        )
                        streamed_result.is_complete = True
                        streamed_result._event_queue.put_nowait(QueueCompleteSentinel())
                    elif isinstance(turn_result.next_step, NextStepRunAgain):
                        continue
                except Exception:
                    cls._handle_streamed_error(streamed_result)
                    raise

            streamed_result.is_complete = True
        except Exception as e:
            streamed_result.is_complete = True
            raise GenericError(e)

    @classmethod
    def _update_streamed_result(
        cls, streamed_result: RunResultStreaming, turn_result: SingleStepResult
    ) -> None:

        streamed_result.raw_responses.append(turn_result.model_response)
        streamed_result.input = turn_result.original_input
        streamed_result.new_items = turn_result.generated_items

    @classmethod
    async def _handle_streamed_final_output(
        cls,
        current_agent: Agent[TContext],
        turn_result: SingleStepResult,
        streamed_result: RunResultStreaming,
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> None:

        streamed_result._output_guardrails_task = asyncio.create_task(
            cls._run_output_guardrails(
                current_agent.output_guardrails + run_config.output_guardrails,
                current_agent,
                turn_result.next_step.output,
                context_wrapper,
            )
        )

        try:
            output_guardrail_results = await streamed_result._output_guardrails_task
        except Exception:
            output_guardrail_results = []

        streamed_result.output_guardrail_results = output_guardrail_results
        streamed_result.final_output = turn_result.next_step.output

    @classmethod
    def _handle_streamed_error(
        cls, streamed_result: RunResultStreaming
    ) -> None:

        streamed_result.is_complete = True
        streamed_result._event_queue.put_nowait(QueueCompleteSentinel())

    @classmethod
    async def _run_input_guardrails_with_queue(
        cls,
        agent: Agent[Any],
        guardrails: list[InputGuardrail[TContext]],
        input: str | list[TResponseInputItem],
        context: RunContextWrapper[TContext],
        streamed_result: RunResultStreaming,
    ):
        queue = streamed_result._input_guardrail_queue

        guardrail_tasks = [
            asyncio.create_task(
                RunImpl.run_single_input_guardrail(agent, guardrail, input, context)
            )
            for guardrail in guardrails
        ]
        guardrail_results = []
        try:
            for done in asyncio.as_completed(guardrail_tasks):
                result = await done
                if result.output.tripwire_triggered:
                    raise InputGuardrailError(result)
                queue.put_nowait(result)
                guardrail_results.append(result)
        except Exception:
            for t in guardrail_tasks:
                t.cancel()
            raise

        streamed_result.input_guardrail_results = guardrail_results

    @classmethod
    async def _run_single_turn_streamed(
        cls,
        streamed_result: RunResultStreaming,
        agent: Agent[TContext],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_hooks: bool,
    ) -> SingleStepResult:

        if should_run_agent_start_hooks:
            await asyncio.gather(
                hooks.on_start(context_wrapper, agent),
                (
                    agent.hooks.on_start(context_wrapper, agent)
                    if agent.hooks
                    else noop_coroutine()
                ),
            )

        output_schema = cls._get_output_schema(agent)
        streamed_result.current_agent = agent
        streamed_result._current_agent_output_schema = output_schema
        system_prompt = await agent.get_system_prompt(context_wrapper)
        handoffs = cls._get_handoffs(agent)
        model = cls._get_model(agent, run_config)
        model_settings = agent.model_settings.resolve(run_config.model_settings)
        final_response: ModelResponse | None = None
        input = ItemHelpers.input_to_new_input_list(streamed_result.input)
        input.extend([item.to_input_item() for item in streamed_result.new_items])

        # 1. Stream the output events
        async for event in model.stream_response(
            system_prompt,
            input,
            model_settings,
            agent.tools,
            output_schema,
            handoffs,
        ):
            if isinstance(event, ResponseCompletedEvent):
                usage = (
                    Usage(
                        requests=1,
                        input_tokens=event.response.usage.input_tokens,
                        output_tokens=event.response.usage.output_tokens,
                        total_tokens=event.response.usage.total_tokens,
                    )
                    if event.response.usage
                    else Usage()
                )
                final_response = ModelResponse(
                    output=event.response.output,
                    usage=usage,
                    referenceable_id=event.response.id,
                )

            streamed_result._event_queue.put_nowait(RawResponsesStreamEvent(data=event))

        # 2. At this point, the streaming is complete for this turn of the agent loop.
        if not final_response:
            raise ModelError("Model did not produce a final response!")

        # 3. Now, we can process the turn as we do in the non-streaming case
        single_step_result = await cls._get_single_step_result_from_response(
            agent=agent,
            original_input=streamed_result.input,
            pre_step_items=streamed_result.new_items,
            new_response=final_response,
            output_schema=output_schema,
            handoffs=handoffs,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
        )
        RunImpl.stream_step_result_to_queue(single_step_result, streamed_result._event_queue)
        return single_step_result

    @classmethod
    async def _run_single_turn(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        generated_items: list[RunItem],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_hooks: bool,
    ) -> SingleStepResult:

        if should_run_agent_start_hooks:
            await asyncio.gather(
                hooks.on_start(context_wrapper, agent),
                (
                    agent.hooks.on_start(context_wrapper, agent)
                    if agent.hooks
                    else noop_coroutine()
                ),
            )

        system_prompt = await agent.get_system_prompt(context_wrapper)
        output_schema = cls._get_output_schema(agent)
        handoffs = cls._get_handoffs(agent)
        input = ItemHelpers.input_to_new_input_list(original_input)
        input.extend([generated_item.to_input_item() for generated_item in generated_items])
        new_response = await cls._get_new_response(
            agent,
            system_prompt,
            input,
            output_schema,
            handoffs,
            context_wrapper,
            run_config,
        )

        return await cls._get_single_step_result_from_response(
            agent=agent,
            original_input=original_input,
            pre_step_items=generated_items,
            new_response=new_response,
            output_schema=output_schema,
            handoffs=handoffs,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
        )

    @classmethod
    async def _get_single_step_result_from_response(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        pre_step_items: list[RunItem],
        new_response: ModelResponse,
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> SingleStepResult:

        processed_response = RunImpl.process_model_response(
            agent=agent,
            response=new_response,
            output_schema=output_schema,
            handoffs=handoffs,
        )
        return await RunImpl.execute_tools_and_side_effects(
            agent=agent,
            original_input=original_input,
            pre_step_items=pre_step_items,
            new_response=new_response,
            processed_response=processed_response,
            output_schema=output_schema,
            hooks=hooks,
            context_wrapper=context_wrapper,
            run_config=run_config,
        )

    @classmethod
    async def _run_input_guardrails(
        cls,
        agent: Agent[Any],
        guardrails: list[InputGuardrail[TContext]],
        input: str | list[TResponseInputItem],
        context: RunContextWrapper[TContext],
    ) -> list[InputGuardrailResult]:
        if not guardrails:
            return []

        guardrail_tasks = [
            asyncio.create_task(
                RunImpl.run_single_input_guardrail(agent, guardrail, input, context)
            )
            for guardrail in guardrails
        ]
        guardrail_results = []
        for done in asyncio.as_completed(guardrail_tasks):
            result = await done
            if result.output.tripwire_triggered:
                # Cancel all guardrail tasks if a tripwire is triggered.
                for t in guardrail_tasks:
                    t.cancel()
                raise InputGuardrailError(result)
            else:
                guardrail_results.append(result)

        return guardrail_results

    @classmethod
    async def _run_output_guardrails(
        cls,
        guardrails: list[OutputGuardrail[TContext]],
        agent: Agent[TContext],
        agent_output: Any,
        context: RunContextWrapper[TContext],
    ) -> list[OutputGuardrailResult]:

        if not guardrails:
            return []
        guardrail_tasks = [
            asyncio.create_task(
                RunImpl.run_single_output_guardrail(guardrail, agent, agent_output, context)
            )
            for guardrail in guardrails
        ]
        guardrail_results = []
        for done in asyncio.as_completed(guardrail_tasks):
            result = await done
            if result.output.tripwire_triggered:
                for t in guardrail_tasks:
                    t.cancel()
                raise OutputGuardrailError(result)
            else:
                guardrail_results.append(result)
        return guardrail_results

    @classmethod
    async def _get_new_response(
        cls,
        agent: Agent[TContext],
        system_prompt: str | None,
        input: list[TResponseInputItem],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> ModelResponse:

        model = cls._get_model(agent, run_config)
        model_settings = agent.model_settings.resolve(run_config.model_settings)
        new_response = await model.get_response(
            system_instructions=system_prompt,
            input=input,
            model_settings=model_settings,
            tools=agent.tools,
            output_schema=output_schema,
            handoffs=handoffs,
        )
        context_wrapper.usage.add(new_response.usage)
        return new_response

    @classmethod
    def _get_output_schema(cls, agent: Agent[Any]) -> AgentOutputSchema | None:
        """Get output schema for agent if specified."""
        if agent.output_type is None or agent.output_type is str:
            return None
        return AgentOutputSchema(agent.output_type)

    @classmethod
    def _get_handoffs(cls, agent: Agent[Any]) -> list[Handoff]:
        """Get list of handoffs from agent, converting Agent instances to Handoff objects."""
        return [
            handoff_item if isinstance(handoff_item, Handoff) else handoff(handoff_item)
            for handoff_item in agent.handoffs
        ]

    @classmethod
    def _get_model(cls, agent: Agent[Any], run_config: RunConfig) -> Model:
        """Get the model instance based on configuration and agent settings."""
        if isinstance(run_config.model, Model):
            return run_config.model
        if isinstance(run_config.model, str):
            return run_config.model_provider.get_model(run_config.model)
        if isinstance(agent.model, Model):
            return agent.model
        return run_config.model_provider.get_model(agent.model)

TContext = TypeVar("TContext")
