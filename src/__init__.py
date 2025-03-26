from __future__ import annotations

from typing import Literal

from .agents.agent import Agent
from .agents.output import AgentOutputSchema
from .agents.run import Runner
from .models.chat_completions import ModelChatCompletionsModel
from .models.interface import Model, ModelProvider
from .models.responses import ModelResponsesModel
from .models.settings import ModelSettings
from .util import _config
from .util._computer import AsyncComputer, Button, Computer, Environment
from .util._exceptions import (
    AgentError,
    InputGuardrailError,
    MaxTurnsError,
    ModelError,
    OutputGuardrailError,
    UsageError,
)
from .util._guardrail import (
    InputGuardrail,
    InputGuardrailResult,
    OutputGuardrail,
    OutputGuardrailResult,
    input_guardrail,
    output_guardrail,
)
from .util._handoffs import Handoff, HandoffInputData, HandoffInputFilter, handoff
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
from .util._types import AsyncDeepSeek
from .util._usage import Usage


def set_default_model_key(key: str) -> None:
    _config.set_default_model_key(key)


def set_default_model_client(client: AsyncDeepSeek) -> None:
    _config.set_default_model_client(client)


def set_default_model_api(api: Literal["chat_completions", "responses"]) -> None:
    _config.set_default_model_api(api)


__all__ = [
    "Agent",
    "Runner",
    "Model",
    "ModelProvider",
    "ModelSettings",
    "ModelChatCompletionsModel",
    "ModelResponsesModel",
    "AgentOutputSchema",
    "Computer",
    "AsyncComputer",
    "Environment",
    "Button",
    "AgentError",
    "InputGuardrailError",
    "OutputGuardrailError",
    "MaxTurnsError",
    "ModelError",
    "UsageError",
    "InputGuardrail",
    "InputGuardrailResult",
    "OutputGuardrail",
    "OutputGuardrailResult",
    "input_guardrail",
    "output_guardrail",
    "handoff",
    "Handoff",
    "HandoffInputData",
    "HandoffInputFilter",
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
    "RunHooks",
    "AgentHooks",
    "RunContextWrapper",
    "TContext",
    "RunResult",
    "RunResultStreaming",
    "RawResponsesStreamEvent",
    "RunItemStreamEvent",
    "AgentUpdatedStreamEvent",
    "StreamEvent",
    "FunctionTool",
    "FunctionToolResult",
    "ComputerTool",
    "FileSearchTool",
    "Tool",
    "WebSearchTool",
    "function_tool",
    "Usage",
    "set_default_model_key",
    "set_default_model_client",
    "set_default_model_api",
    "default_tool_error_function",
]
