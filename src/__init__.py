from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .util._types import AsyncDeepSeek

if TYPE_CHECKING:
    from .agents.agent import Agent
    from .agents.output import AgentOutputSchema
    from .agents.run import Runner
    from .gear.orbs import Handoff, HandoffInputData, HandoffInputFilter, handoff
    from .gear.shields import (
        InputShield,
        InputShieldResult,
        OutputShield,
        OutputShieldResult,
        input_shield,
        output_shield,
    )
    from .models.chat import ModelChatCompletionsModel
    from .models.interface import Model, ModelProvider
    from .models.responses import ModelResponsesModel
    from .models.settings import ModelSettings
    from .util._computer import AsyncComputer, Button, Computer, Environment
    from .util._exceptions import (
        AgentError,
        InputShieldError,
        MaxTurnsError,
        ModelError,
        OutputShieldError,
        UsageError,
    )
    from .util._items import (
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
    from .util._lifecycle import AgentHooks, RunHooks
    from .util._result import RunResult, RunResultStreaming
    from .util._run_context import RunContextWrapper, TContext
    from .util._stream_events import (
        AgentUpdatedStreamEvent,
        RawResponsesStreamEvent,
        RunItemStreamEvent,
        StreamEvent,
    )
    from .util._tool import (
        ComputerTool,
        FileSearchTool,
        FunctionTool,
        FunctionToolResult,
        Tool,
        WebSearchTool,
        default_tool_error_function,
        function_tool,
    )
    from .util._usage import Usage


def set_default_model_key(key: str) -> None:
    from .models import shared

    shared.set_default_model_key(key)


def set_default_model_client(client: AsyncDeepSeek) -> None:
    from .models import shared

    shared.set_default_model_client(client)


def set_default_model_api(api: Literal["chat_completions", "responses"]) -> None:
    from .models import shared

    shared.set_use_responses_by_default(api != "chat_completions")


__all__ = [
    # Core components
    "Agent",
    "Runner",
    "Model",
    "ModelProvider",
    "ModelSettings",
    "ModelChatCompletionsModel",
    "ModelResponsesModel",
    "AgentOutputSchema",
    # Computer and environment
    "Computer",
    "AsyncComputer",
    "Environment",
    "Button",
    # Exceptions
    "AgentError",
    "InputShieldError",
    "OutputShieldError",
    "MaxTurnsError",
    "ModelError",
    "UsageError",
    # Shields
    "InputShield",
    "InputShieldResult",
    "OutputShield",
    "OutputShieldResult",
    "input_shield",
    "output_shield",
    # Handoffs
    "handoff",
    "Handoff",
    "HandoffInputData",
    "HandoffInputFilter",
    # Items
    "TResponseInputItem",
    "MessageOutputItem",
    "ModelResponse",
    "RunItem",
    "HandoffCallItem",
    "HandoffOutputItem",
    "ToolCallItem",
    "ToolCallOutputItem",
    "ReasoningItem",
    "ItemHelpers",
    # Lifecycle and context
    "RunHooks",
    "AgentHooks",
    "RunContextWrapper",
    "TContext",
    # Results and events
    "RunResult",
    "RunResultStreaming",
    "RawResponsesStreamEvent",
    "RunItemStreamEvent",
    "AgentUpdatedStreamEvent",
    "StreamEvent",
    # Tools
    "FunctionTool",
    "FunctionToolResult",
    "ComputerTool",
    "FileSearchTool",
    "Tool",
    "WebSearchTool",
    "function_tool",
    "default_tool_error_function",
    # Usage and configuration
    "Usage",
    "set_default_model_key",
    "set_default_model_client",
    "set_default_model_api",
]
