from __future__ import annotations

import abc
import copy
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar, Union, cast

from pydantic import BaseModel
from typing_extensions import TypeAlias

from ._exceptions import ModelError
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


########################################################
#            Data Classes for Outputs                  #
########################################################

@dataclass(frozen=True)
class RunItemBase(Generic[T], abc.ABC):
    """Base class for agent run items.

    Args:
        agent: The agent instance associated with this item
        raw_item: The raw item data
    """
    agent: Agent[Any]
    raw_item: T
    _cached_input_item: TResponseInputItem | None = field(default=None, init=False)

    @cached_property
    def input_item(self) -> TResponseInputItem:
        """Convert and cache item to model input format."""
        if self._cached_input_item is None:
            if isinstance(self.raw_item, dict):
                self._cached_input_item = self.raw_item
            elif isinstance(self.raw_item, BaseModel):
                self._cached_input_item = self.raw_item.model_dump(exclude_unset=True)
            else:
                self._cached_input_item = self.raw_item
        return self._cached_input_item

    def to_input_item(self) -> TResponseInputItem:
        """Convert item to model input format.

        Returns:
            The item converted to input format
        """
        return self.input_item


@dataclass(frozen=True)
class MessageOutputItem(RunItemBase[ResponseOutputItem]):
    """LLM message output."""
    raw_item: ResponseOutputItem
    type: Literal["message_output_item"] = "message_output_item"

    @cached_property
    def text_content(self) -> str:
        """Cache the text content of the message."""
        return "".join(
            item["text"]
            for item in self.raw_item["content"]
            if item["type"] == OUTPUT_TEXT_TYPE
        )


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


ToolCallItemTypes: TypeAlias = Union[
    ResponseFunctionToolCall,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionWebSearch,
]


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


RunItem: TypeAlias = Union[
    MessageOutputItem,
    HandoffCallItem,
    HandoffOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    ReasoningItem,
]


@dataclass(frozen=True)
class ModelResponse:
    """Model response containing outputs and usage information.

    Args:
        output: List of response output items
        usage: Usage statistics for the response
        referenceable_id: Optional ID for referencing this response
    """
    output: list[TResponseOutputItem]
    usage: Usage
    referenceable_id: str | None
    _cached_input_items: list[TResponseInputItem] | None = field(default=None, init=False)

    @cached_property
    def input_items(self) -> list[TResponseInputItem]:
        """Convert and cache outputs to input format."""
        if self._cached_input_items is None:
            self._cached_input_items = [
                cast(TResponseInputItem, it.model_dump(exclude_unset=True))
                for it in self.output
            ]
        return self._cached_input_items

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
        if not hasattr(message, "type") or message.type != MESSAGE_TYPE:
            return ""

        last_content = message.content[-1]
        if last_content.type == OUTPUT_TEXT_TYPE:
            return last_content.text
        if last_content.type == REFUSAL_TYPE:
            return last_content.refusal
        raise ModelError(f"Unexpected content type: {last_content.type}")

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
        input: str | list[TResponseInputItem]
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
        """Extract text from a message output efficiently."""
        text = message.text_content
        if text.endswith("', 'type': 'output_text', 'annotations': []}"):
            text = text[:-len("', 'type': 'output_text', 'annotations': []}")]
        return text.strip()

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
        if content.endswith("', 'type': 'output_text', 'annotations': []}"):
            content = content[:-len("', 'type': 'output_text', 'annotations': []}")]
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return ""

        formatted_lines = []
        current_section = []

        def format_section(section: list[str]) -> list[str]:
            if not section:
                return []
            width = max(len(l) for l in section)
            border = "+" + "-" * (width + 2) + "+"
            return [border, *[f"| {l:<{width}} |" for l in section], border]

        for line in lines:
            if line.startswith(ItemHelpers.SPECIAL_LINES):
                if current_section:
                    formatted_lines.extend(format_section(current_section))
                    current_section = []
                if line.startswith(ECHOES_START):
                    formatted_lines.append('')
                formatted_lines.append(line)
            elif line.endswith(THINK_END):
                formatted_lines.append(line)
            else:
                current_section.append(line)

        if current_section:
            formatted_lines.extend(format_section(current_section))

        return '\n'.join(formatted_lines)
