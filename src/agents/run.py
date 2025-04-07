"""
The Runner class is the core orchestrator for executing agents in the Noctira system.
It provides both synchronous and asynchronous interfaces for running agents with various
configurations and capabilities.

Key features:
- Supports both synchronous (run_sync) and asynchronous (run) execution
- Provides streaming capabilities (run_streamed) for real-time agent responses
- Handles input and output shields for security and validation
- Manages agent charms for custom behavior hooks
- Supports multiple turns of agent execution with configurable limits
- Integrates with various model providers and settings
- Handles complex input/output processing with orbs and schemas
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from ..gear.charm import RunCharms
from ..gear.orbs import Orbs, OrbsInputFilter, orbs
from ..gear.shield import (
    InputShield,
    InputShieldResult,
    OutputShield,
    OutputShieldResult,
)
from ..models.interface import Model
from ..models.provider import ModelProvider
from ..models.settings import ModelSettings
from ..util._constants import MAX_TURNS
from ..util._exceptions import (
    AgentError,
    GenericError,
    InputShieldError,
    MaxTurnsError,
    ModelError,
    OutputShieldError,
)
from ..util._items import ItemHelpers, ModelResponse, RunItem
from ..util._result import RunResult, RunResultStreaming
from ..util._stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent
from ..util._types import ResponseEvent, RunContextWrapper, TContext, TResponseInputItem, Usage
from .agent import Agent
from .output import AgentOutputSchema
from .run_impl import (
    NextStepFinalOutput,
    NextStepOrbs,
    NextStepRunAgain,
    QueueCompleteSentinel,
    RunImpl,
    SingleStepResult,
)

########################################################
#               Data class for Run Config
########################################################


@dataclass(frozen=True)
class RunConfig:
    model: str | Model | None = None
    model_provider: ModelProvider = field(default_factory=ModelProvider)
    model_settings: ModelSettings | None = None
    orbs_input_filter: OrbsInputFilter | None = None
    input_shields: list[InputShield[TContext]] = field(default_factory=list)
    output_shields: list[OutputShield[TContext]] = field(default_factory=list)
    max_turns: int = MAX_TURNS


########################################################
#               Main Class: Runner
########################################################


class Runner:
    @classmethod
    async def _initialize_run(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        context: TContext | None,
        max_turns: int,
        charms: RunCharms[TContext] | None,
        run_config: RunConfig | None,
    ) -> tuple[
        RunCharms[TContext],
        RunConfig,
        RunContextWrapper[TContext],
        str | list[TResponseInputItem],
        AgentOutputSchema | None,
    ]:
        charms = charms or RunCharms[Any]()
        run_config = run_config or RunConfig()
        config_dict = {k: v for k, v in run_config.__dict__.items() if k != "max_turns"}
        return (
            charms,
            RunConfig(**config_dict, max_turns=max_turns),
            RunContextWrapper(context=context),
            input,
            cls._get_output_schema(starting_agent),
        )

    @classmethod
    async def run(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = MAX_TURNS,
        charms: RunCharms[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResult:
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
                    charms,
                    context_wrapper,
                    run_config,
                    should_run_agent_start_charms,
                    input,
                )

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
                    raise AgentError(f"Unknown next step type: {type(turn_result.next_step)}")
        except Exception as e:
            raise GenericError(e) from e

    @classmethod
    async def _run_turn(
        cls,
        current_turn: int,
        current_agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        generated_items: list[RunItem],
        charms: RunCharms[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_charms: bool,
        input: str | list[TResponseInputItem],
    ) -> SingleStepResult:
        if current_turn == 1:
            _, turn_result = await asyncio.gather(
                cls._run_input_shields(
                    agent=current_agent,
                    input=input,
                    context=context_wrapper,
                    shields=current_agent.input_shields + run_config.input_shields,
                ),
                cls._run_single_turn(
                    agent=current_agent,
                    original_input=original_input,
                    generated_items=generated_items,
                    charms=charms,
                    context_wrapper=context_wrapper,
                    run_config=run_config,
                    should_run_agent_start_charms=should_run_agent_start_charms,
                ),
            )
        else:
            turn_result = await cls._run_single_turn(
                agent=current_agent,
                original_input=original_input,
                generated_items=generated_items,
                charms=charms,
                context_wrapper=context_wrapper,
                run_config=run_config,
                should_run_agent_start_charms=should_run_agent_start_charms,
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
        input_shield_results: list[InputShieldResult],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> RunResult:
        output_shield_results = await cls._run_output_shields(
            current_agent,
            turn_result.next_step.output,
            context_wrapper,
            current_agent.output_shields + run_config.output_shields,
        )
        return RunResult(
            input=original_input,
            new_items=generated_items,
            raw_responses=model_responses,
            final_output=turn_result.next_step.output,
            _last_agent=current_agent,
            input_shield_results=input_shield_results,
            output_shield_results=output_shield_results,
        )

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        *,
        context: TContext | None = None,
        max_turns: int = MAX_TURNS,
        charms: RunCharms[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResult:
        """Synchronous wrapper for run()."""
        return asyncio.get_event_loop().run_until_complete(
            cls.run(
                starting_agent,
                input,
                context=context,
                max_turns=max_turns,
                charms=charms,
                run_config=run_config,
            )
        )

    @classmethod
    async def run_streamed(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        context: TContext | None = None,
        max_turns: int = MAX_TURNS,
        charms: RunCharms[TContext] | None = None,
        run_config: RunConfig | None = None,
    ) -> RunResultStreaming:
        charms, run_config, context_wrapper, input, output_schema = await cls._initialize_run(
            starting_agent, input, context, max_turns, charms, run_config
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
                charms=charms,
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
        return RunResultStreaming(
            input=input if isinstance(input, str) else input.copy(),
            new_items=[],
            current_agent=starting_agent,
            raw_responses=[],
            final_output=None,
            is_complete=False,
            current_turn=0,
            max_turns=max_turns,
            input_shield_results=[],
            output_shield_results=[],
        )

    @classmethod
    async def _run_streamed_impl(
        cls,
        starting_input: str | list[TResponseInputItem],
        streamed_result: RunResultStreaming,
        starting_agent: Agent[TContext],
        max_turns: int,
        charms: RunCharms[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> None:
        current_agent = starting_agent
        current_turn = 0
        should_run_agent_start_charms = True

        try:
            await cls._queue_event(
                streamed_result._event_queue,
                AgentUpdatedStreamEvent(new_agent=current_agent),
            )
        except (asyncio.TimeoutError, asyncio.QueueFull):
            pass

        try:
            while not streamed_result.is_complete:
                current_turn += 1
                streamed_result.current_turn = current_turn

                if current_turn > max_turns:
                    await cls._queue_event(streamed_result._event_queue, QueueCompleteSentinel())
                    break

                if current_turn == 1:
                    streamed_result._input_shields_task = asyncio.create_task(
                        cls._run_input_shields_with_queue(
                            agent=starting_agent,
                            input=starting_input,
                            context=context_wrapper,
                            shields=starting_agent.input_shields + run_config.input_shields,
                            streamed_result=streamed_result,
                        )
                    )

                try:
                    turn_result = await cls._run_single_turn_streamed(
                        streamed_result,
                        current_agent,
                        charms,
                        context_wrapper,
                        run_config,
                        should_run_agent_start_charms,
                    )
                    should_run_agent_start_charms = False

                    cls._update_streamed_result(streamed_result, turn_result)

                    if isinstance(turn_result.next_step, NextStepOrbs):
                        current_agent = turn_result.next_step.new_agent
                        should_run_agent_start_charms = True
                        try:
                            await asyncio.wait_for(
                                streamed_result._event_queue.put(
                                    AgentUpdatedStreamEvent(new_agent=current_agent)
                                ),
                                timeout=1.0,
                            )
                        except (asyncio.TimeoutError, asyncio.QueueFull):
                            pass
                    elif isinstance(turn_result.next_step, NextStepFinalOutput):
                        await cls._handle_streamed_final_output(
                            current_agent,
                            turn_result,
                            streamed_result,
                            context_wrapper,
                            run_config,
                        )
                        streamed_result.is_complete = True
                        try:
                            await asyncio.wait_for(
                                streamed_result._event_queue.put(QueueCompleteSentinel()),
                                timeout=1.0,
                            )
                        except (asyncio.TimeoutError, asyncio.QueueFull):
                            pass
                    elif isinstance(turn_result.next_step, NextStepRunAgain):
                        continue
                except Exception:
                    cls._handle_streamed_error(streamed_result)
                    raise

            streamed_result.is_complete = True
        except Exception as e:
            streamed_result.is_complete = True
            raise GenericError(e) from e

    @classmethod
    async def _queue_event(cls, queue: asyncio.Queue, event: Any) -> None:
        try:
            await asyncio.wait_for(queue.put(event), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.QueueFull):
            pass

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
        streamed_result._output_shields_task = asyncio.create_task(
            cls._run_output_shields(
                current_agent,
                turn_result.next_step.output,
                context_wrapper,
                current_agent.output_shields + run_config.output_shields,
            )
        )

        try:
            output_shield_results = await streamed_result._output_shields_task
        except Exception:
            output_shield_results = []

        streamed_result.output_shield_results = output_shield_results
        streamed_result.final_output = turn_result.next_step.output

    @classmethod
    def _handle_streamed_error(cls, streamed_result: RunResultStreaming) -> None:
        streamed_result.is_complete = True
        asyncio.create_task(cls._queue_event(streamed_result._event_queue, QueueCompleteSentinel()))

    @classmethod
    async def _run_input_shields_with_queue(
        cls,
        agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        context: RunContextWrapper[TContext],
        shields: list[InputShield[TContext]],
        streamed_result: RunResultStreaming,
    ):
        if not shields:
            return
        input_list = ItemHelpers.input_to_new_input_list(input)
        input_to_use = deepcopy(input_list) if len(shields) > 1 else input_list

        shield_tasks = [
            asyncio.create_task(
                RunImpl.run_single_input_shield(agent, shield, input_to_use, context)
            )
            for shield in shields
        ]
        shield_results = []
        try:
            for done in asyncio.as_completed(shield_tasks):
                result = await done
                if result.output.tripwire_triggered:
                    raise InputShieldError(result)
                streamed_result._input_shield_queue.put_nowait(result)
                shield_results.append(result)
        except Exception:
            for t in shield_tasks:
                t.cancel()
            raise

        streamed_result.input_shield_results = shield_results

    @classmethod
    async def _run_single_turn_streamed(
        cls,
        streamed_result: RunResultStreaming,
        agent: Agent[TContext],
        charms: RunCharms[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_charms: bool,
    ) -> SingleStepResult:
        if should_run_agent_start_charms:
            await cls._run_agent_start_charms(
                charms,
                agent,
                context_wrapper,
            )

        system_prompt = await agent.get_system_prompt(context_wrapper)
        input = streamed_result.input
        output_schema = cls._get_output_schema(agent)
        orbs = cls._get_orbs(agent)
        model = cls._get_model(agent, run_config)
        model_settings = agent.model_settings.resolve(run_config.model_settings)

        # Stream the output events and collect the final response
        async for event in model.stream_response(
            system_prompt,
            input,
            model_settings,
            agent.swords,
            output_schema,
            orbs=orbs,
        ):
            if isinstance(event, ResponseEvent) and event.type == "completed":
                usage = (
                    Usage(
                        requests=1,
                        input_tokens=event.response.usage.input_tokens,
                        output_tokens=event.response.usage.output_tokens,
                        total_tokens=event.response.usage.total_tokens,
                    )
                    if event.response.usage
                    else Usage(
                        requests=1,
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                    )
                )
                final_response = ModelResponse(
                    output=event.response.output,
                    usage=usage,
                    referenceable_id=event.response.id,
                )
                # Process the response immediately when we get it
                processed_response = RunImpl.process_model_response(
                    agent=agent,
                    response=final_response,
                    output_schema=output_schema,
                    orbs=orbs,
                )

                single_step_result = await RunImpl.execute_swords_and_side_effects(
                    agent=agent,
                    original_input=streamed_result.input,
                    pre_step_items=streamed_result.new_items,
                    new_response=final_response,
                    processed_response=processed_response,
                    output_schema=output_schema,
                    charms=charms,
                    context_wrapper=context_wrapper,
                    run_config=run_config,
                )

                # Queue the step result events
                await RunImpl.stream_step_result_to_queue(
                    single_step_result, streamed_result._event_queue
                )
                return single_step_result

            # Queue raw response events
            await cls._queue_event(
                streamed_result._event_queue, RawResponsesStreamEvent(data=event)
            )

        raise ModelError("Model did not produce a final response!")

    @classmethod
    async def _run_single_turn(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        generated_items: list[RunItem],
        charms: RunCharms[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
        should_run_agent_start_charms: bool,
    ) -> SingleStepResult:
        if should_run_agent_start_charms:
            await cls._run_agent_start_charms(
                charms,
                agent,
                context_wrapper,
            )

        system_prompt = await agent.get_system_prompt(context_wrapper)
        output_schema = cls._get_output_schema(agent)
        orbs = cls._get_orbs(agent)
        input = ItemHelpers.input_to_new_input_list(original_input)
        input.extend([generated_item.to_input_item() for generated_item in generated_items])
        new_response = await cls._get_new_response(
            agent,
            system_prompt,
            input,
            output_schema,
            orbs,
            context_wrapper,
            run_config,
        )

        return await cls._get_single_step_result_from_response(
            agent=agent,
            original_input=original_input,
            pre_step_items=generated_items,
            new_response=new_response,
            output_schema=output_schema,
            orbs=orbs,
            charms=charms,
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
        orbs: list[Orbs],
        charms: RunCharms[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> SingleStepResult:
        processed_response = RunImpl.process_model_response(
            agent=agent,
            response=new_response,
            output_schema=output_schema,
            orbs=orbs,
        )
        return await RunImpl.execute_swords_and_side_effects(
            agent=agent,
            original_input=original_input,
            pre_step_items=pre_step_items,
            new_response=new_response,
            processed_response=processed_response,
            output_schema=output_schema,
            charms=charms,
            context_wrapper=context_wrapper,
            run_config=run_config,
        )

    @classmethod
    async def _run_input_shields(
        cls,
        agent: Agent[TContext],
        input: str | list[TResponseInputItem],
        context: RunContextWrapper[TContext],
        shields: list[InputShield[TContext]],
    ) -> list[InputShieldResult]:
        if not shields:
            return []

        shield_tasks = [
            asyncio.create_task(RunImpl.run_single_input_shield(agent, shield, input, context))
            for shield in shields
        ]

        shield_results = []
        for done in asyncio.as_completed(shield_tasks):
            result = await done
            # Cancel all shield tasks if a tripwire is triggered.
            for t in shield_tasks:
                if not t.done():
                    t.cancel()
            shield_results.append(result)

        return shield_results

    @classmethod
    async def _run_output_shields(
        cls,
        agent: Agent[TContext],
        agent_output: Any,
        context: RunContextWrapper[TContext],
        shields: list[OutputShield[TContext]],
    ) -> list[OutputShieldResult]:
        if not shields:
            return []

        shield_tasks = [
            asyncio.create_task(
                RunImpl.run_single_output_shield(shield, agent, agent_output, context)
            )
            for shield in shields
        ]

        shield_results = []
        for done in asyncio.as_completed(shield_tasks):
            result = await done
            if result.output.tripwire_triggered:
                raise OutputShieldError(result)
            for t in shield_tasks:
                if not t.done():
                    t.cancel()
            shield_results.append(result)
        return shield_results

    @classmethod
    async def _get_new_response(
        cls,
        agent: Agent[TContext],
        system_prompt: str | None,
        input: list[TResponseInputItem],
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> ModelResponse:
        model = cls._get_model(agent, run_config)
        model_settings = agent.model_settings.resolve(run_config.model_settings)
        new_response = await model.get_response(
            system_instructions=system_prompt,
            input=input,
            model_settings=model_settings,
            swords=agent.swords,
            output_schema=output_schema,
            orbs=orbs,
        )
        context_wrapper.usage.add(new_response.usage)
        return new_response

    @classmethod
    def _get_output_schema(cls, agent: Agent[Any]) -> AgentOutputSchema | None:
        if agent.output_type is None or agent.output_type is str:
            return None
        return AgentOutputSchema(agent.output_type)

    @classmethod
    def _get_orbs(cls, agent: Agent[Any]) -> list[Orbs]:
        return [
            orbs_item if isinstance(orbs_item, Orbs) else orbs(orbs_item)
            for orbs_item in agent.orbs
        ]

    @classmethod
    def _get_model(cls, agent: Agent[Any], run_config: RunConfig) -> Model:
        if isinstance(run_config.model, Model):
            return run_config.model
        if isinstance(run_config.model, str):
            return run_config.model_provider.get_model(run_config.model)
        if isinstance(agent.model, Model):
            return agent.model
        return run_config.model_provider.get_model(agent.model)

    @classmethod
    async def _run_agent_start_charms(
        cls,
        charms: RunCharms[TContext],
        agent: Agent[TContext],
        context_wrapper: RunContextWrapper[TContext],
    ) -> None:
        charm_tasks = [
            charms.on_start(context_wrapper, agent),
            agent.charms.on_start(context_wrapper, agent) if agent.charms else None,
        ]
        await asyncio.gather(*[t for t in charm_tasks if t is not None])
