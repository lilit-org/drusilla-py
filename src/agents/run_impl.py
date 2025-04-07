"""Core implementation for agent execution and run management.

This module provides the core implementation for executing agent runs, handling the flow of agent
interactions, and processing various types of responses and actions. It manages:

- Processing model responses and converting them into actionable items
- Executing function calls (swords) and handling their results
- Managing agent transitions (orbs) between different agent instances
- Handling input and output shielding for security and validation
- Streaming run events and managing the execution flow
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..gear.charm import RunCharms
from ..gear.orbs import Orbs, OrbsInputData
from ..gear.shield import (
    InputShield,
    InputShieldResult,
    OutputShield,
    OutputShieldResult,
)
from ..gear.sword import Sword, SwordResult
from ..util._constants import logger
from ..util._exceptions import AgentError, ModelError, UsageError
from ..util._items import (
    ItemHelpers,
    MessageOutputItem,
    ModelResponse,
    OrbsCallItem,
    OrbsOutputItem,
    ReasoningItem,
    RunItem,
    SwordCallItem,
    SwordCallOutputItem,
)
from ..util._stream_events import RunItemStreamEvent, StreamEvent
from ..util._types import (
    InputItem,
    QueueCompleteSentinel,
    ResponseFunctionSwordCall,
    RunContextWrapper,
    TContext,
)
from .agent import Agent, SwordsToFinalOutputResult
from .output import AgentOutputSchema

if TYPE_CHECKING:
    from .run import RunConfig


########################################################
#               Data Classes for Swords
########################################################


@dataclass(frozen=True)
class SwordRunOrbs:
    orbs: Orbs
    sword_call: ResponseFunctionSwordCall


@dataclass(frozen=True)
class SwordRunFunction:
    sword_call: ResponseFunctionSwordCall
    function_sword: Sword


@dataclass
class ProcessedResponse:
    new_items: list[RunItem]
    orbs: list[SwordRunOrbs]
    functions: list[SwordRunFunction]

    def has_swords_to_run(self) -> bool:
        return bool(self.orbs or self.functions)


@dataclass(frozen=True)
class NextStepOrbs:
    new_agent: Agent[Any]


@dataclass(frozen=True)
class NextStepFinalOutput:
    output: Any


@dataclass(frozen=True)
class NextStepRunAgain:
    pass


@dataclass
class SingleStepResult:
    original_input: str | list[InputItem]
    model_response: ModelResponse
    pre_step_items: list[RunItem]
    new_step_items: list[RunItem]
    next_step: NextStepOrbs | NextStepFinalOutput | NextStepRunAgain

    @property
    def generated_items(self) -> list[RunItem]:
        return self.pre_step_items + self.new_step_items


########################################################
#               Main Class: RunImpl
########################################################


class RunImpl:
    EVENT_MAP = {
        MessageOutputItem: "message_output_created",
        OrbsCallItem: "orbs_requested",
        OrbsOutputItem: "orbs_occured",
        SwordCallItem: "sword_called",
        SwordCallOutputItem: "sword_output",
        ReasoningItem: "reasoning_item_created",
    }

    _NOT_FINAL_OUTPUT = SwordsToFinalOutputResult(is_final_output=False, final_output=None)

    @classmethod
    async def execute_swords_and_side_effects(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[InputItem],
        pre_step_items: list[RunItem],
        new_response: ModelResponse,
        processed_response: ProcessedResponse,
        output_schema: AgentOutputSchema | None,
        charms: RunCharms[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> SingleStepResult:
        new_step_items = processed_response.new_items

        function_results = await cls.execute_function_sword_calls(
            agent=agent,
            sword_runs=processed_response.functions,
            charms=charms,
            context_wrapper=context_wrapper,
        )
        new_step_items.extend([result.run_item for result in function_results])

        if run_orbs := processed_response.orbs:
            return await cls.execute_orbs(
                agent=agent,
                original_input=original_input,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                new_response=new_response,
                run_orbs=run_orbs,
                charms=charms,
                context_wrapper=context_wrapper,
                run_config=run_config,
            )

        check_sword_use = await cls._check_for_final_output_from_swords(
            agent=agent,
            sword_results=function_results,
            context_wrapper=context_wrapper,
        )

        if check_sword_use.is_final_output:
            if not agent.output_type or agent.output_type is str:
                check_sword_use.final_output = str(check_sword_use.final_output)

            if check_sword_use.final_output is None:
                logger.error("Model returned None.")

            return await cls.execute_final_output(
                agent=agent,
                original_input=original_input,
                new_response=new_response,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                final_output=check_sword_use.final_output,
                charms=charms,
                context_wrapper=context_wrapper,
            )

        message_items = [item for item in new_step_items if isinstance(item, MessageOutputItem)]
        potential_final_output_text = (
            ItemHelpers.extract_last_text(message_items[-1].raw_item) if message_items else None
        )

        if output_schema and not output_schema.is_plain_text() and potential_final_output_text:
            final_output = output_schema.validate_json(potential_final_output_text)
            return await cls.execute_final_output(
                agent=agent,
                original_input=original_input,
                new_response=new_response,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                final_output=final_output,
                charms=charms,
                context_wrapper=context_wrapper,
            )
        elif (
            not output_schema or output_schema.is_plain_text()
        ) and not processed_response.has_swords_to_run():
            return await cls.execute_final_output(
                agent=agent,
                original_input=original_input,
                new_response=new_response,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                final_output=potential_final_output_text or "",
                charms=charms,
                context_wrapper=context_wrapper,
            )
        else:
            return SingleStepResult(
                original_input=original_input,
                model_response=new_response,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                next_step=NextStepRunAgain(),
            )

    @classmethod
    def process_model_response(
        cls,
        *,
        agent: Agent[Any],
        response: ModelResponse,
        output_schema: AgentOutputSchema | None,
        orbs: list[Orbs],
    ) -> ProcessedResponse:
        items: list[RunItem] = []
        run_orbs: list[SwordRunOrbs] = []
        functions: list[SwordRunFunction] = []
        orbs_map = {orb.sword_name: orb for orb in orbs}
        function_map = {}

        for sword in agent.swords:
            if isinstance(sword, Sword):
                function_map[sword.name] = sword

        for output in response.output:
            output_type = output["type"]
            if output_type == "message":
                items.append(MessageOutputItem(raw_item=output, agent=agent))
            elif output_type == "output_text":
                message_output = {
                    "type": "message",
                    "content": [output],
                    "role": "assistant",
                }
                items.append(MessageOutputItem(raw_item=message_output, agent=agent))
            elif output_type in ("file_search", "web_search", "interface"):
                items.append(SwordCallItem(raw_item=output, agent=agent))
            elif output_type == "reasoning":
                items.append(ReasoningItem(raw_item=output, agent=agent))
            elif output_type == "function":
                if output["name"] in orbs_map:
                    items.append(OrbsCallItem(raw_item=output, agent=agent))
                    orbs = SwordRunOrbs(
                        orbs=orbs_map[output["name"]],
                        sword_call=output,
                    )
                    run_orbs.append(orbs)
                else:
                    if output["name"] not in function_map:
                        raise ModelError(f"Sword {output['name']} not found in agent {agent.name}")
                    items.append(SwordCallItem(raw_item=output, agent=agent))
                    functions.append(
                        SwordRunFunction(
                            sword_call=output,
                            function_sword=function_map[output["name"]],
                        )
                    )
            else:
                logger.warning(f"Unexpected output type, ignoring: {output_type}")
                continue

        return ProcessedResponse(
            new_items=items,
            orbs=run_orbs,
            functions=functions,
        )

    @classmethod
    async def execute_function_sword_calls(
        cls,
        *,
        agent: Agent[TContext],
        sword_runs: list[SwordRunFunction],
        charms: RunCharms[TContext],
        context_wrapper: RunContextWrapper[TContext],
    ) -> list[SwordResult]:
        async def run_single_sword(func_sword: Sword, sword_call: ResponseFunctionSwordCall) -> Any:
            try:
                # Run charms and sword in parallel
                charms_tasks = [
                    charms.on_sword_start(context_wrapper, agent, func_sword),
                    func_sword.on_invoke_sword(context_wrapper, sword_call["arguments"]),
                ]
                if agent.charms:
                    charms_tasks.insert(
                        1, agent.charms.on_sword_start(context_wrapper, agent, func_sword)
                    )

                results = await asyncio.gather(*[t for t in charms_tasks if t is not None])
                result = results[-1]  # The last result is from on_invoke_sword

                # Run end charms in parallel
                end_charms_tasks = [charms.on_sword_end(context_wrapper, agent, func_sword, result)]
                if agent.charms:
                    end_charms_tasks.append(
                        agent.charms.on_sword_end(context_wrapper, agent, func_sword, result)
                    )

                await asyncio.gather(*[t for t in end_charms_tasks if t is not None])
                return result
            except Exception as e:
                raise AgentError(e) from e

        # Run all swords in parallel
        results = await asyncio.gather(
            *[
                run_single_sword(sword_run.function_sword, sword_run.sword_call)
                for sword_run in sword_runs
            ]
        )

        return [
            SwordResult(
                sword=sword_run.function_sword,
                output=result,
                run_item=SwordCallOutputItem(
                    output=result,
                    raw_item=ItemHelpers.sword_call_output_item(sword_run.sword_call, str(result)),
                    agent=agent,
                ),
            )
            for sword_run, result in zip(sword_runs, results, strict=False)
        ]

    @classmethod
    async def execute_orbs(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[InputItem],
        pre_step_items: list[RunItem],
        new_step_items: list[RunItem],
        new_response: ModelResponse,
        run_orbs: list[SwordRunOrbs],
        charms: RunCharms[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig | None,
    ) -> SingleStepResult:
        # Handle multiple orbs case
        if len(run_orbs) > 1:
            ignored_orbs = run_orbs[1:]
            output_message = f"Multiple orbs, ignoring {len(ignored_orbs)} orbs."
            new_step_items.append(
                SwordCallOutputItem(
                    output=output_message,
                    raw_item=ItemHelpers.sword_call_output_item(
                        ignored_orbs[0].sword_call, output_message
                    ),
                    agent=agent,
                )
            )

        # Process the first orb
        actual_orbs = run_orbs[0]
        orbs = actual_orbs.orbs
        new_agent: Agent[Any] = await orbs.on_invoke_orbs(
            context_wrapper, actual_orbs.sword_call["arguments"]
        )

        # Add orb output item
        new_step_items.append(
            OrbsOutputItem(
                agent=agent,
                raw_item=ItemHelpers.sword_call_output_item(
                    actual_orbs.sword_call,
                    orbs.get_transfer_message(new_agent),
                ),
                source_agent=agent,
                target_agent=new_agent,
            )
        )

        # Run charms in parallel
        charms_tasks = [
            charms.on_orbs(context=context_wrapper, from_agent=agent, to_agent=new_agent),
            (
                agent.charms.on_orbs(context_wrapper, agent=new_agent, source=agent)
                if agent.charms
                else None
            ),
        ]
        await asyncio.gather(*[t for t in charms_tasks if t is not None])

        # Apply input filter if present
        input_filter = orbs.input_filter or (run_config.orbs_input_filter if run_config else None)
        if input_filter:
            if not callable(input_filter):
                raise UsageError(f"Invalid input filter: {input_filter}")

            orbs_input_data = OrbsInputData(
                input_history=(
                    tuple(original_input) if isinstance(original_input, list) else original_input
                ),
                pre_orbs_items=tuple(pre_step_items),
                new_items=tuple(new_step_items),
            )

            filtered = input_filter(orbs_input_data)
            if not isinstance(filtered, OrbsInputData):
                raise UsageError(f"Invalid input filter result: {filtered}")

            original_input = (
                filtered.input_history
                if isinstance(filtered.input_history, str)
                else list(filtered.input_history)
            )
            pre_step_items = list(filtered.pre_orbs_items)
            new_step_items = list(filtered.new_items)

        return SingleStepResult(
            original_input=original_input,
            model_response=new_response,
            pre_step_items=pre_step_items,
            new_step_items=new_step_items,
            next_step=NextStepOrbs(new_agent),
        )

    @classmethod
    async def execute_final_output(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[InputItem],
        new_response: ModelResponse,
        pre_step_items: list[RunItem],
        new_step_items: list[RunItem],
        final_output: Any,
        charms: RunCharms[TContext],
        context_wrapper: RunContextWrapper[TContext],
    ) -> SingleStepResult:
        await cls.run_final_output_charms(agent, charms, context_wrapper, final_output)

        return SingleStepResult(
            original_input=original_input,
            model_response=new_response,
            pre_step_items=pre_step_items,
            new_step_items=new_step_items,
            next_step=NextStepFinalOutput(final_output),
        )

    @classmethod
    async def run_final_output_charms(
        cls,
        agent: Agent[TContext],
        charms: RunCharms[TContext],
        context_wrapper: RunContextWrapper[TContext],
        final_output: Any,
    ):
        if agent.charms:
            await asyncio.gather(
                charms.on_end(context_wrapper, agent, final_output),
                agent.charms.on_end(context_wrapper, agent, final_output),
            )
        else:
            await charms.on_end(context_wrapper, agent, final_output)

    @classmethod
    async def run_single_input_shield(
        cls,
        agent: Agent[Any],
        shield: InputShield[TContext],
        input: str | list[InputItem],
        context: RunContextWrapper[TContext],
    ) -> InputShieldResult:
        return await shield.run(agent, input, context)

    @classmethod
    async def run_single_output_shield(
        cls,
        shield: OutputShield[TContext],
        agent: Agent[Any],
        agent_output: Any,
        context: RunContextWrapper[TContext],
    ) -> OutputShieldResult:
        return await shield.run(
            context=context,
            agent=agent,
            agent_output=agent_output,
        )

    @classmethod
    async def stream_step_result_to_queue(
        cls,
        step_result: SingleStepResult,
        queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel],
    ):
        events = [
            RunItemStreamEvent(item=item, name=cls.EVENT_MAP[type(item)])
            for item in step_result.new_step_items
            if type(item) in cls.EVENT_MAP
        ]

        # Batch put all events with timeout
        for event in events:
            try:
                await asyncio.wait_for(queue.put(event), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.QueueFull):
                logger.warning(f"Failed to queue event: {event}")

    @classmethod
    async def _check_for_final_output_from_swords(
        cls,
        *,
        agent: Agent[TContext],
        sword_results: list[SwordResult],
        context_wrapper: RunContextWrapper[TContext],
    ) -> SwordsToFinalOutputResult:
        if not sword_results:
            return cls._NOT_FINAL_OUTPUT

        if agent.sword_use_behavior == "run_llm_again":
            return cls._NOT_FINAL_OUTPUT
        elif agent.sword_use_behavior == "stop_on_first_sword":
            return SwordsToFinalOutputResult(
                is_final_output=True, final_output=sword_results[0].output
            )
        elif isinstance(agent.sword_use_behavior, dict):
            names = agent.sword_use_behavior.get("stop_at_sword_names", [])
            for sword_result in sword_results:
                if sword_result.sword.name in names:
                    return SwordsToFinalOutputResult(
                        is_final_output=True, final_output=sword_result.output
                    )
            return cls._NOT_FINAL_OUTPUT
        elif callable(agent.sword_use_behavior):
            if inspect.iscoroutinefunction(agent.sword_use_behavior):
                if result := await cast(
                    Awaitable[SwordsToFinalOutputResult],
                    agent.sword_use_behavior(context_wrapper, sword_results),
                ):
                    return result
            else:
                if result := cast(
                    SwordsToFinalOutputResult,
                    agent.sword_use_behavior(context_wrapper, sword_results),
                ):
                    return result

        logger.error(f"Invalid sword_use_behavior: {agent.sword_use_behavior}")
        raise UsageError(f"Invalid sword_use_behavior: {agent.sword_use_behavior}")
