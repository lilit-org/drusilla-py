from __future__ import annotations

import abc
import copy
from dataclasses import dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel
from typing_extensions import TypeAlias

from ._types import (
    ComputerCallOutput,
    FunctionCallOutput,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseReasoningItem,
    ResponseStreamEvent,
)
from ._usage import Usage

if TYPE_CHECKING:
    from ..agents.agent import Agent


########################################################
#            Type Aliases and Constants                #
########################################################

TResponseInputItem = ResponseInputItemParam
TResponseOutputItem = ResponseOutputItem
TResponseStreamEvent = ResponseStreamEvent
T = TypeVar("T", bound=Union[TResponseOutputItem, TResponseInputItem])

MESSAGE_TYPE = "message"
OUTPUT_TEXT_TYPE = "output_text"
REFUSAL_TYPE = "refusal"
THINK_START = "<think>"
THINK_END = "</think>"
ECHOES_START = "Echoes of encrypted hearts"

# Type aliases for tool calls
ToolCallItemTypes: TypeAlias = Union[
    ResponseFunctionToolCall,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionWebSearch,
]

# Type alias for all possible run items
RunItem: TypeAlias = Union[
    "MessageOutputItem",
    "HandoffCallItem",
    "HandoffOutputItem",
    "ToolCallItem",
    "ToolCallOutputItem",
    "ReasoningItem",
]


@dataclass(frozen=True)
class RunItemBase(Generic[T], abc.ABC):
    """Base class for agent run items."""

    agent: Agent[Any]
    raw_item: T

    @cached_property
    def input_item(self) -> TResponseInputItem:
        """Convert and cache item to model input format."""
        if isinstance(self.raw_item, dict):
            return self.raw_item
        elif isinstance(self.raw_item, BaseModel):
            return self.raw_item.model_dump(exclude_unset=True)
        return self.raw_item

    def to_input_item(self) -> TResponseInputItem:
        """Convert item to model input format."""
        return self.input_item


@dataclass(frozen=True)
class MessageOutputItem(RunItemBase[ResponseOutputItem]):
    """LLM message output."""

    raw_item: ResponseOutputItem
    type: Literal["message_output_item"] = "message_output_item"

    @cached_property
    def text_content(self) -> str:
        """Cache the text content of the message."""
        try:
            if not isinstance(self.raw_item, dict):
                return ""

            content = self.raw_item.get("content", [])
            if not content:
                return ""

            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == OUTPUT_TEXT_TYPE:
                        text = item.get("text", "")
                        if text:
                            texts.append(text)
                elif hasattr(item, "type") and item.type == OUTPUT_TEXT_TYPE:
                    text = getattr(item, "text", "")
                    if text:
                        texts.append(text)

            return " ".join(texts).strip()

        except (AttributeError, KeyError, TypeError, IndexError):
            return ""


@dataclass(frozen=True)
class HandoffCallItem(RunItemBase[ResponseFunctionToolCall]):
    """Agent handoff tool call."""

    raw_item: ResponseFunctionToolCall
    type: Literal["handoff_call_item"] = "handoff_call_item"


@dataclass(frozen=True)
class HandoffOutputItem(RunItemBase[TResponseInputItem]):
    """Agent handoff output."""

    raw_item: TResponseInputItem
    source_agent: Agent[Any]
    target_agent: Agent[Any]
    type: Literal["handoff_output_item"] = "handoff_output_item"


@dataclass(frozen=True)
class ToolCallItem(RunItemBase[ToolCallItemTypes]):
    """Tool call for function or computer action."""

    raw_item: ToolCallItemTypes
    type: Literal["tool_call_item"] = "tool_call_item"


@dataclass(frozen=True)
class ToolCallOutputItem(RunItemBase[Union[FunctionCallOutput, ComputerCallOutput]]):
    """Tool call execution output."""

    raw_item: FunctionCallOutput | ComputerCallOutput
    output: Any
    type: Literal["tool_call_output_item"] = "tool_call_output_item"


@dataclass(frozen=True)
class ReasoningItem(RunItemBase[ResponseReasoningItem]):
    """LLM reasoning step."""

    raw_item: ResponseReasoningItem
    type: Literal["reasoning_item"] = "reasoning_item"


@dataclass(frozen=True)
class ModelResponse:
    """Model response containing outputs and usage information."""

    output: list[TResponseOutputItem]
    usage: Usage
    referenceable_id: str | None

    @cached_property
    def input_items(self) -> list[TResponseInputItem]:
        """Convert and cache outputs to input format."""
        return [
            cast(TResponseInputItem, it.model_dump(exclude_unset=True))
            for it in self.output
        ]

    def to_input_items(self) -> list[TResponseInputItem]:
        """Convert outputs to input format efficiently."""
        return self.input_items


########################################################
#                   Item Helpers                        #
########################################################


class ItemHelpers:
    """Helper methods for processing and formatting various item types."""

    SPECIAL_LINES: ClassVar[tuple[str, str]] = (THINK_START, ECHOES_START)

    @staticmethod
    def extract_last_content(message: TResponseOutputItem) -> str:
        """Extract the last content from a message."""
        try:
            if not hasattr(message, "type") or message.type != MESSAGE_TYPE:
                return ""

            if not hasattr(message, "content") or not message.content:
                return ""

            last_content = message.content[-1]
            if last_content.type == OUTPUT_TEXT_TYPE:
                return last_content.text
            if last_content.type == REFUSAL_TYPE:
                return last_content.refusal
            return ""
        except (AttributeError, IndexError, KeyError):
            return ""

    @staticmethod
    def extract_last_text(message: TResponseOutputItem) -> str | None:
        """Extract the last text content from a message, ignoring refusals."""
        if hasattr(message, "type") and message.type == MESSAGE_TYPE:
            last_content = message.content[-1]
            if last_content.type == OUTPUT_TEXT_TYPE:
                return last_content.text
        return None

    @staticmethod
    def input_to_new_input_list(
        input: str | list[TResponseInputItem],
    ) -> list[TResponseInputItem]:
        """Convert string or input items to input list."""
        if isinstance(input, str):
            return [{"content": input, "role": "user"}]
        return copy.deepcopy(input)

    @staticmethod
    def text_message_outputs(items: list[RunItem]) -> str:
        """Concatenate text from all message outputs efficiently."""
        return "".join(
            item.text_content if isinstance(item, MessageOutputItem) else ""
            for item in items
        ).strip()

    @staticmethod
    def text_message_output(message: MessageOutputItem) -> str:
        """Extract and format text from a message output efficiently."""
        try:
            if not (text := message.text_content):
                return ""

            # Clean up the raw text
            text = text.replace("', 'type': 'output_text', 'annotations': []}", "")
            text = text.replace("'text': '", "")
            text = text.replace("'", "")
            text = text.strip()
            if not text:
                return ""

            # Split into lines and process
            lines = []
            current_line = []

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Handle think tags
                if line.startswith(THINK_START):
                    if current_line:
                        lines.append(" ".join(current_line))
                        current_line = []
                    lines.append(line)
                elif line.endswith(THINK_END):
                    if current_line:
                        lines.append(" ".join(current_line))
                        current_line = []
                    lines.append(line)
                else:
                    current_line.append(line)

            # Add any remaining lines
            if current_line:
                lines.append(" ".join(current_line))

            return "\n".join(lines)

        except (IndexError, KeyError, AttributeError):
            return ""

    @staticmethod
    def tool_call_output_item(
        tool_call: ResponseFunctionToolCall, output: str
    ) -> FunctionCallOutput:
        """Create a tool call output from a call and result."""
        return {
            "call_id": tool_call.call_id,
            "output": output,
            "type": "function_call_output",
        }

    @staticmethod
    def format_content(content: str) -> str:
        """Format content with proper indentation and borders efficiently."""
        if not content:
            return ""
        content = content.replace("', 'type': 'output_text', 'annotations': []}", "")
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if not lines:
            return ""

        formatted_lines = []
        current_section = []

        def format_section(section: list[str]) -> list[str]:
            if not section:
                return []
            width = max(len(line) for line in section)
            border = "+" + "-" * (width + 2) + "+"
            return [border, *[f"| {line:<{width}} |" for line in section], border]

        for line in lines:
            if line.startswith(ItemHelpers.SPECIAL_LINES):
                if current_section:
                    formatted_lines.extend(format_section(current_section))
                    current_section = []
                if line.startswith(ECHOES_START):
                    formatted_lines.append("")
                formatted_lines.append(line)
            elif line.endswith(THINK_END):
                formatted_lines.append(line)
            else:
                current_section.append(line)

        if current_section:
            formatted_lines.extend(format_section(current_section))

        return "\n".join(formatted_lines)
