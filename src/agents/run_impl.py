from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..gear.orbs import Orbs, OrbsInputData
from ..gear.shields import (
    InputShield,
    InputShieldResult,
    OutputShield,
    OutputShieldResult,
)
from ..gear.swords import ComputerSword, FunctionSword, FunctionSwordResult
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
    TResponseInputItem,
)
from ..util._lifecycle import RunHooks
from ..util._logger import logger
from ..util._run_context import RunContextWrapper, TContext
from ..util._stream_events import RunItemStreamEvent, StreamEvent
from ..util._types import (
    ComputerAction,
    ResponseComputerSwordCall,
    ResponseFunctionSwordCall,
)
from .agent import Agent, SwordsToFinalOutputResult
from .output import AgentOutputSchema

if TYPE_CHECKING:
    from .run import RunConfig


########################################################
#               Constants
########################################################


async def noop_coroutine() -> None:
    """A coroutine that does nothing. Used as a fallback when no hooks are defined."""
    return None


class QueueCompleteSentinel:
    pass


QUEUE_COMPLETE_SENTINEL = QueueCompleteSentinel()
_NOT_FINAL_OUTPUT = SwordsToFinalOutputResult(is_final_output=False, final_output=None)


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
    function_sword: FunctionSword


@dataclass(frozen=True)
class SwordRunComputerAction:
    sword_call: ResponseComputerSwordCall
    computer_sword: ComputerSword


@dataclass
class ProcessedResponse:
    new_items: list[RunItem]
    orbs: list[SwordRunOrbs]
    functions: list[SwordRunFunction]
    computer_actions: list[SwordRunComputerAction]

    def has_swords_to_run(self) -> bool:
        return bool(self.orbs or self.functions or self.computer_actions)


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
    original_input: str | list[TResponseInputItem]
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

    @classmethod
    async def execute_swords_and_side_effects(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        pre_step_items: list[RunItem],
        new_response: ModelResponse,
        processed_response: ProcessedResponse,
        output_schema: AgentOutputSchema | None,
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> SingleStepResult:
        new_step_items = processed_response.new_items

        function_results, computer_results = await asyncio.gather(
            cls.execute_function_sword_calls(
                agent=agent,
                sword_runs=processed_response.functions,
                hooks=hooks,
                context_wrapper=context_wrapper,
            ),
            cls.execute_computer_actions(
                agent=agent,
                actions=processed_response.computer_actions,
                hooks=hooks,
                context_wrapper=context_wrapper,
            ),
        )
        new_step_items.extend([result.run_item for result in function_results])
        new_step_items.extend(computer_results)

        if run_orbs := processed_response.orbs:
            return await cls.execute_orbs(
                agent=agent,
                original_input=original_input,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                new_response=new_response,
                run_orbs=run_orbs,
                hooks=hooks,
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
                hooks=hooks,
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
                hooks=hooks,
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
                hooks=hooks,
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
        computer_actions: list[SwordRunComputerAction] = []
        orbs_map = {orb.sword_name: orb for orb in orbs}
        function_map = {}
        computer_sword = None

        for sword in agent.swords:
            if isinstance(sword, FunctionSword):
                function_map[sword.name] = sword
            elif isinstance(sword, ComputerSword):
                computer_sword = sword

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
            elif output_type in ("file_search", "web_search", "computer"):
                items.append(SwordCallItem(raw_item=output, agent=agent))
                if output_type == "computer":
                    if not computer_sword:
                        raise ModelError("Model produced computer action without a computer sword.")
                    computer_actions.append(
                        SwordRunComputerAction(sword_call=output, computer_sword=computer_sword)
                    )
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
            computer_actions=computer_actions,
        )

    @classmethod
    async def execute_function_sword_calls(
        cls,
        *,
        agent: Agent[TContext],
        sword_runs: list[SwordRunFunction],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
    ) -> list[FunctionSwordResult]:
        async def run_single_sword(
            func_sword: FunctionSword, sword_call: ResponseFunctionSwordCall
        ) -> Any:
            try:
                _, _, result = await asyncio.gather(
                    hooks.on_sword_start(context_wrapper, agent, func_sword),
                    (
                        agent.hooks.on_sword_start(context_wrapper, agent, func_sword)
                        if agent.hooks
                        else noop_coroutine()
                    ),
                    func_sword.on_invoke_sword(context_wrapper, sword_call.arguments),
                )

                await asyncio.gather(
                    hooks.on_sword_end(context_wrapper, agent, func_sword, result),
                    (
                        agent.hooks.on_sword_end(context_wrapper, agent, func_sword, result)
                        if agent.hooks
                        else noop_coroutine()
                    ),
                )
            except Exception as e:
                raise AgentError(e) from e

            return result

        results = await asyncio.gather(
            *[
                run_single_sword(sword_run.function_sword, sword_run.sword_call)
                for sword_run in sword_runs
            ]
        )

        return [
            FunctionSwordResult(
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
    async def execute_computer_actions(
        cls,
        *,
        agent: Agent[TContext],
        actions: list[SwordRunComputerAction],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
    ) -> list[RunItem]:
        return await asyncio.gather(
            *[
                ComputerAction.execute(
                    agent=agent,
                    action=action,
                    hooks=hooks,
                    context_wrapper=context_wrapper,
                )
                for action in actions
            ]
        )

    @classmethod
    async def execute_orbs(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        pre_step_items: list[RunItem],
        new_step_items: list[RunItem],
        new_response: ModelResponse,
        run_orbs: list[SwordRunOrbs],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig | None,
    ) -> SingleStepResult:
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

        actual_orbs = run_orbs[0]
        orbs = actual_orbs.orbs
        new_agent: Agent[Any] = await orbs.on_invoke_orbs(
            context_wrapper, actual_orbs.sword_call.arguments
        )

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

        await asyncio.gather(
            hooks.on_orbs(
                context=context_wrapper,
                from_agent=agent,
                to_agent=new_agent,
            ),
            (
                agent.hooks.on_orbs(
                    context_wrapper,
                    agent=new_agent,
                    source=agent,
                )
                if agent.hooks
                else noop_coroutine()
            ),
        )

        input_filter = orbs.input_filter or (run_config.orbs_input_filter if run_config else None)
        if input_filter:
            logger.debug("Filtering inputs for orbs")
            orbs_input_data = OrbsInputData(
                input_history=(
                    tuple(original_input) if isinstance(original_input, list) else original_input
                ),
                pre_orbs_items=tuple(pre_step_items),
                new_items=tuple(new_step_items),
            )
            if not callable(input_filter):
                raise UsageError(f"Invalid input filter: {input_filter}")
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
        original_input: str | list[TResponseInputItem],
        new_response: ModelResponse,
        pre_step_items: list[RunItem],
        new_step_items: list[RunItem],
        final_output: Any,
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
    ) -> SingleStepResult:
        await cls.run_final_output_hooks(agent, hooks, context_wrapper, final_output)

        return SingleStepResult(
            original_input=original_input,
            model_response=new_response,
            pre_step_items=pre_step_items,
            new_step_items=new_step_items,
            next_step=NextStepFinalOutput(final_output),
        )

    @classmethod
    async def run_final_output_hooks(
        cls,
        agent: Agent[TContext],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        final_output: Any,
    ):
        if agent.hooks:
            await asyncio.gather(
                hooks.on_end(context_wrapper, agent, final_output),
                agent.hooks.on_end(context_wrapper, agent, final_output),
            )
        else:
            await hooks.on_end(context_wrapper, agent, final_output)

    @classmethod
    async def run_single_input_shield(
        cls,
        agent: Agent[Any],
        shield: InputShield[TContext],
        input: str | list[TResponseInputItem],
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
        events = []
        for item in step_result.new_step_items:
            event_name = cls.EVENT_MAP.get(type(item))
            if event_name:
                events.append(RunItemStreamEvent(item=item, name=event_name))
            else:
                logger.warning(f"Unexpected item type: {type(item)}")

        # Batch put all events at once with timeout
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
        sword_results: list[FunctionSwordResult],
        context_wrapper: RunContextWrapper[TContext],
    ) -> SwordsToFinalOutputResult:
        if not sword_results:
            return _NOT_FINAL_OUTPUT

        if agent.sword_use_behavior == "run_llm_again":
            return _NOT_FINAL_OUTPUT
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
            return _NOT_FINAL_OUTPUT
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
