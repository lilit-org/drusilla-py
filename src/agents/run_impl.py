from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..util._computer import AsyncComputer, Computer
from ..util._coro import noop_coroutine as noop
from ..util._exceptions import AgentError, ModelError, UsageError
from ..util._guardrail import (
    InputGuardrail,
    InputGuardrailResult,
    OutputGuardrail,
    OutputGuardrailResult,
)
from ..util._handoffs import Handoff, HandoffInputData
from ..util._items import (
    HandoffCallItem,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    ModelResponse,
    ReasoningItem,
    RunItem,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
)
from ..util._lifecycle import RunHooks
from ..util._logger import logger
from ..util._run_context import RunContextWrapper, TContext
from ..util._stream_events import RunItemStreamEvent, StreamEvent
from ..util._tool import ComputerTool, FunctionTool, FunctionToolResult
from ..util._types import (
    ComputerAction,
    ResponseComputerToolCall,
    ResponseFunctionToolCall,
)
from .agent import Agent, ToolsToFinalOutputResult
from .output import AgentOutputSchema

if TYPE_CHECKING:
    from .run import RunConfig


########################################################
#               Constants                                #
########################################################

class QueueCompleteSentinel:
    pass

QUEUE_COMPLETE_SENTINEL = QueueCompleteSentinel()
_NOT_FINAL_OUTPUT = ToolsToFinalOutputResult(is_final_output=False, final_output=None)


########################################################
#               Data Classes                            #
########################################################

@dataclass
class ToolRunHandoff:
    handoff: Handoff
    tool_call: ResponseFunctionToolCall


@dataclass
class ToolRunFunction:
    tool_call: ResponseFunctionToolCall
    function_tool: FunctionTool


@dataclass
class ToolRunComputerAction:
    tool_call: ResponseComputerToolCall
    computer_tool: ComputerTool


@dataclass
class ProcessedResponse:
    new_items: list[RunItem]
    handoffs: list[ToolRunHandoff]
    functions: list[ToolRunFunction]
    computer_actions: list[ToolRunComputerAction]

    def has_tools_to_run(self) -> bool:
        return bool(self.handoffs or self.functions or self.computer_actions)


@dataclass
class NextStepHandoff:
    new_agent: Agent[Any]


@dataclass
class NextStepFinalOutput:
    output: Any


@dataclass
class NextStepRunAgain:
    pass


@dataclass
class SingleStepResult:
    original_input: str | list[TResponseInputItem]
    model_response: ModelResponse
    pre_step_items: list[RunItem]
    new_step_items: list[RunItem]
    next_step: NextStepHandoff | NextStepFinalOutput | NextStepRunAgain

    @property
    def generated_items(self) -> list[RunItem]:
        return self.pre_step_items + self.new_step_items


########################################################
#               Main Class: RunImpl                    #
########################################################

class RunImpl:
    @classmethod
    async def execute_tools_and_side_effects(
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
        pre_step_items = list(pre_step_items)
        new_step_items = list(processed_response.new_items)

        function_results, computer_results = await asyncio.gather(
            cls.execute_function_tool_calls(
                agent=agent,
                tool_runs=processed_response.functions,
                hooks=hooks,
                context_wrapper=context_wrapper,
            ),
            cls.execute_computer_actions(
                agent=agent,
                actions=processed_response.computer_actions,
                hooks=hooks,
                context_wrapper=context_wrapper,
                config=run_config,
            ),
        )
        new_step_items.extend([result.run_item for result in function_results])
        new_step_items.extend(computer_results)

        if run_handoffs := processed_response.handoffs:
            return await cls.execute_handoffs(
                agent=agent,
                original_input=original_input,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                new_response=new_response,
                run_handoffs=run_handoffs,
                hooks=hooks,
                context_wrapper=context_wrapper,
                run_config=run_config,
            )

        check_tool_use = await cls._check_for_final_output_from_tools(
            agent=agent,
            tool_results=function_results,
            context_wrapper=context_wrapper,
        )

        if check_tool_use.is_final_output:
            if not agent.output_type or agent.output_type is str:
                check_tool_use.final_output = str(check_tool_use.final_output)

            if check_tool_use.final_output is None:
                logger.error("Model returned a final output of None. Not raising an error because we assume you know what you're doing.")

            return await cls.execute_final_output(
                agent=agent,
                original_input=original_input,
                new_response=new_response,
                pre_step_items=pre_step_items,
                new_step_items=new_step_items,
                final_output=check_tool_use.final_output,
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
        ) and not processed_response.has_tools_to_run():
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
        handoffs: list[Handoff],
    ) -> ProcessedResponse:
        items: list[RunItem] = []
        run_handoffs = []
        functions = []
        computer_actions = []

        handoff_map = {handoff.tool_name: handoff for handoff in handoffs}
        function_map = {tool.name: tool for tool in agent.tools if isinstance(tool, FunctionTool)}
        computer_tool = next((tool for tool in agent.tools if isinstance(tool, ComputerTool)), None)

        for output in response.output:
            output_type = output["type"]
            if output_type == "message":
                items.append(MessageOutputItem(raw_item=output, agent=agent))
            elif output_type == "output_text":
                message_output = {
                    "type": "message",
                    "content": [output],
                    "role": "assistant"
                }
                items.append(MessageOutputItem(raw_item=message_output, agent=agent))
            elif output_type in ("file_search", "web_search", "computer"):
                items.append(ToolCallItem(raw_item=output, agent=agent))
                if output_type == "computer" and not computer_tool:
                    raise ModelError("Model produced computer action without a computer tool.")
                if output_type == "computer":
                    computer_actions.append(
                        ToolRunComputerAction(tool_call=output, computer_tool=computer_tool)
                    )
            elif output_type == "reasoning":
                items.append(ReasoningItem(raw_item=output, agent=agent))
            elif output_type == "function":
                if output["name"] in handoff_map:
                    items.append(HandoffCallItem(raw_item=output, agent=agent))
                    handoff = ToolRunHandoff(
                        tool_call=output,
                        handoff=handoff_map[output["name"]],
                    )
                    run_handoffs.append(handoff)
                else:
                    if output["name"] not in function_map:
                        raise ModelError(f"Tool {output['name']} not found in agent {agent.name}")
                    items.append(ToolCallItem(raw_item=output, agent=agent))
                    functions.append(
                        ToolRunFunction(
                            tool_call=output,
                            function_tool=function_map[output["name"]],
                        )
                    )
            else:
                logger.warning(f"Unexpected output type, ignoring: {output_type}")
                continue

        return ProcessedResponse(
            new_items=items,
            handoffs=run_handoffs,
            functions=functions,
            computer_actions=computer_actions,
        )

    @classmethod
    async def execute_function_tool_calls(
        cls,
        *,
        agent: Agent[TContext],
        tool_runs: list[ToolRunFunction],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
    ) -> list[FunctionToolResult]:
        async def run_single_tool(
            func_tool: FunctionTool, tool_call: ResponseFunctionToolCall
        ) -> Any:
            try:
                _, _, result = await asyncio.gather(
                    hooks.on_tool_start(context_wrapper, agent, func_tool),
                    (
                        agent.hooks.on_tool_start(context_wrapper, agent, func_tool)
                        if agent.hooks
                        else noop()
                    ),
                    func_tool.on_invoke_tool(context_wrapper, tool_call.arguments),
                )

                await asyncio.gather(
                    hooks.on_tool_end(context_wrapper, agent, func_tool, result),
                    (
                        agent.hooks.on_tool_end(context_wrapper, agent, func_tool, result)
                        if agent.hooks
                        else noop()
                    ),
                )
            except Exception as e:
                raise AgentError(e)

            return result

        results = await asyncio.gather(*[run_single_tool(tool_run.function_tool, tool_run.tool_call) for tool_run in tool_runs])

        return [
            FunctionToolResult(
                tool=tool_run.function_tool,
                output=result,
                run_item=ToolCallOutputItem(
                    output=result,
                    raw_item=ItemHelpers.tool_call_output_item(tool_run.tool_call, str(result)),
                    agent=agent,
                ),
            )
            for tool_run, result in zip(tool_runs, results)
        ]

    @classmethod
    async def execute_computer_actions(
        cls,
        *,
        agent: Agent[TContext],
        actions: list[ToolRunComputerAction],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        config: RunConfig,
    ) -> list[RunItem]:
        results: list[RunItem] = []
        for action in actions:
            results.append(
                await ComputerAction.execute(
                    agent=agent,
                    action=action,
                    hooks=hooks,
                    context_wrapper=context_wrapper,
                )
            )
        return results

    @classmethod
    async def execute_handoffs(
        cls,
        *,
        agent: Agent[TContext],
        original_input: str | list[TResponseInputItem],
        pre_step_items: list[RunItem],
        new_step_items: list[RunItem],
        new_response: ModelResponse,
        run_handoffs: list[ToolRunHandoff],
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
        run_config: RunConfig,
    ) -> SingleStepResult:
        if len(run_handoffs) > 1:
            output_message = "Multiple handoffs detected, ignoring this one."
            new_step_items.extend(
                [
                    ToolCallOutputItem(
                        output=output_message,
                        raw_item=ItemHelpers.tool_call_output_item(
                            handoff.tool_call, output_message
                        ),
                        agent=agent,
                    )
                    for handoff in run_handoffs[1:]
                ]
            )

        actual_handoff = run_handoffs[0]
        handoff = actual_handoff.handoff
        new_agent: Agent[Any] = await handoff.on_invoke_handoff(
            context_wrapper, actual_handoff.tool_call.arguments
        )

        new_step_items.append(
            HandoffOutputItem(
                agent=agent,
                raw_item=ItemHelpers.tool_call_output_item(
                    actual_handoff.tool_call,
                    handoff.get_transfer_message(new_agent),
                ),
                source_agent=agent,
                target_agent=new_agent,
            )
        )

        await asyncio.gather(
            hooks.on_handoff(
                context=context_wrapper,
                from_agent=agent,
                to_agent=new_agent,
            ),
            (
                agent.hooks.on_handoff(
                    context_wrapper,
                    agent=new_agent,
                    source=agent,
                )
                if agent.hooks
                else noop()
            ),
        )

        input_filter = handoff.input_filter or (
            run_config.handoff_input_filter if run_config else None
        )
        if input_filter:
            logger.debug("Filtering inputs for handoff")
            handoff_input_data = HandoffInputData(
                input_history=tuple(original_input)
                if isinstance(original_input, list)
                else original_input,
                pre_handoff_items=tuple(pre_step_items),
                new_items=tuple(new_step_items),
            )
            if not callable(input_filter):
                raise UsageError(f"Invalid input filter: {input_filter}")
            filtered = input_filter(handoff_input_data)
            if not isinstance(filtered, HandoffInputData):
                raise UsageError(f"Invalid input filter result: {filtered}")

            original_input = (
                filtered.input_history
                if isinstance(filtered.input_history, str)
                else list(filtered.input_history)
            )
            pre_step_items = list(filtered.pre_handoff_items)
            new_step_items = list(filtered.new_items)

        return SingleStepResult(
            original_input=original_input,
            model_response=new_response,
            pre_step_items=pre_step_items,
            new_step_items=new_step_items,
            next_step=NextStepHandoff(new_agent),
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
        await asyncio.gather(
            hooks.on_end(context_wrapper, agent, final_output),
            agent.hooks.on_end(context_wrapper, agent, final_output)
            if agent.hooks
            else noop(),
        )

    @classmethod
    async def run_single_input_guardrail(
        cls,
        agent: Agent[Any],
        guardrail: InputGuardrail[TContext],
        input: str | list[TResponseInputItem],
        context: RunContextWrapper[TContext],
    ) -> InputGuardrailResult:
        return await guardrail.run(agent, input, context)

    @classmethod
    async def run_single_output_guardrail(
        cls,
        guardrail: OutputGuardrail[TContext],
        agent: Agent[Any],
        agent_output: Any,
        context: RunContextWrapper[TContext],
    ) -> OutputGuardrailResult:
        return await guardrail.run(agent=agent, agent_output=agent_output, context=context)

    @classmethod
    def stream_step_result_to_queue(
        cls,
        step_result: SingleStepResult,
        queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel],
    ):
        for item in step_result.new_step_items:
            if isinstance(item, MessageOutputItem):
                event = RunItemStreamEvent(item=item, name="message_output_created")
            elif isinstance(item, HandoffCallItem):
                event = RunItemStreamEvent(item=item, name="handoff_requested")
            elif isinstance(item, HandoffOutputItem):
                event = RunItemStreamEvent(item=item, name="handoff_occured")
            elif isinstance(item, ToolCallItem):
                event = RunItemStreamEvent(item=item, name="tool_called")
            elif isinstance(item, ToolCallOutputItem):
                event = RunItemStreamEvent(item=item, name="tool_output")
            elif isinstance(item, ReasoningItem):
                event = RunItemStreamEvent(item=item, name="reasoning_item_created")
            else:
                logger.warning(f"Unexpected item type: {type(item)}")
                event = None

            if event:
                queue.put_nowait(event)

    @classmethod
    async def _check_for_final_output_from_tools(
        cls,
        *,
        agent: Agent[TContext],
        tool_results: list[FunctionToolResult],
        context_wrapper: RunContextWrapper[TContext],
    ) -> ToolsToFinalOutputResult:
        if not tool_results:
            return _NOT_FINAL_OUTPUT

        if agent.tool_use_behavior == "run_llm_again":
            return _NOT_FINAL_OUTPUT
        elif agent.tool_use_behavior == "stop_on_first_tool":
            return ToolsToFinalOutputResult(
                is_final_output=True, final_output=tool_results[0].output
            )
        elif isinstance(agent.tool_use_behavior, dict):
            names = agent.tool_use_behavior.get("stop_at_tool_names", [])
            for tool_result in tool_results:
                if tool_result.tool.name in names:
                    return ToolsToFinalOutputResult(
                        is_final_output=True, final_output=tool_result.output
                    )
            return _NOT_FINAL_OUTPUT
        elif callable(agent.tool_use_behavior):
            if inspect.iscoroutinefunction(agent.tool_use_behavior):
                return await cast(
                    Awaitable[ToolsToFinalOutputResult],
                    agent.tool_use_behavior(context_wrapper, tool_results),
                )
            else:
                return cast(
                    ToolsToFinalOutputResult, agent.tool_use_behavior(context_wrapper, tool_results)
                )

        logger.error(f"Invalid tool_use_behavior: {agent.tool_use_behavior}")
        raise UsageError(f"Invalid tool_use_behavior: {agent.tool_use_behavior}")


########################################################
#               Main Class: Computer Action            #
########################################################

class ComputerAction:
    @classmethod
    async def execute(
        cls,
        *,
        agent: Agent[TContext],
        action: ToolRunComputerAction,
        hooks: RunHooks[TContext],
        context_wrapper: RunContextWrapper[TContext],
    ) -> RunItem:
        output_func = (
            cls._get_screenshot_async(action.computer_tool.computer, action.tool_call)
            if isinstance(action.computer_tool.computer, AsyncComputer)
            else cls._get_screenshot_sync(action.computer_tool.computer, action.tool_call)
        )

        _, _, output = await asyncio.gather(
            hooks.on_tool_start(context_wrapper, agent, action.computer_tool),
            (
                agent.hooks.on_tool_start(context_wrapper, agent, action.computer_tool)
                if agent.hooks
                else noop()
            ),
            output_func,
        )

        await asyncio.gather(
            hooks.on_tool_end(context_wrapper, agent, action.computer_tool, output),
            (
                agent.hooks.on_tool_end(context_wrapper, agent, action.computer_tool, output)
                if agent.hooks
                else noop()
            ),
        )

        image_url = f"data:image/png;base64,{output}"
        return ToolCallOutputItem(
            agent=agent,
            output=image_url,
            raw_item={
                "call_id": action.tool_call["call_id"],
                "output": {
                    "type": "computer_screenshot",
                    "image_url": image_url,
                },
                "type": "computer_call_output",
            },
        )

    @classmethod
    async def _get_screenshot_sync(
        cls,
        computer: Computer,
        tool_call: ResponseComputerToolCall,
    ) -> str:
        action = tool_call["action"]
        if action["type"] == "click":
            computer.click(action["x"], action["y"], action["button"])
        elif action["type"] == "double_click":
            computer.double_click(action["x"], action["y"])
        elif action["type"] == "drag":
            computer.drag([(p["x"], p["y"]) for p in action["path"]])
        elif action["type"] == "keypress":
            computer.keypress(action["keys"])
        elif action["type"] == "move":
            computer.move(action["x"], action["y"])
        elif action["type"] == "screenshot":
            computer.screenshot()
        elif action["type"] == "scroll":
            computer.scroll(action["x"], action["y"], action["scroll_x"], action["scroll_y"])
        elif action["type"] == "type":
            computer.type(action["text"])
        elif action["type"] == "wait":
            computer.wait()

        return computer.screenshot()

    @classmethod
    async def _get_screenshot_async(
        cls,
        computer: AsyncComputer,
        tool_call: ResponseComputerToolCall,
    ) -> str:
        action = tool_call["action"]
        if action["type"] == "click":
            await computer.click(action["x"], action["y"], action["button"])
        elif action["type"] == "double_click":
            await computer.double_click(action["x"], action["y"])
        elif action["type"] == "drag":
            await computer.drag([(p["x"], p["y"]) for p in action["path"]])
        elif action["type"] == "keypress":
            await computer.keypress(action["keys"])
        elif action["type"] == "move":
            await computer.move(action["x"], action["y"])
        elif action["type"] == "screenshot":
            await computer.screenshot()
        elif action["type"] == "scroll":
            await computer.scroll(action["x"], action["y"], action["scroll_x"], action["scroll_y"])
        elif action["type"] == "type":
            await computer.type(action["text"])
        elif action["type"] == "wait":
            await computer.wait()

        return await computer.screenshot()
