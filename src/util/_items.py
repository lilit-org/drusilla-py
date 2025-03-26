from __future__ import annotations

import abc
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, Union

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


########################################################
#            Data Classes for Outputs                  #
########################################################

@dataclass
class RunItemBase(Generic[T], abc.ABC):
    """Base class for agent run items.

    Args:
        agent: The agent instance associated with this item
        raw_item: The raw item data
    """
    agent: Agent[Any]
    raw_item: T

    def to_input_item(self) -> TResponseInputItem:
        """Convert item to model input format.

        Returns:
            The item converted to input format
        """
        if isinstance(self.raw_item, dict):
            return self.raw_item
        elif isinstance(self.raw_item, BaseModel):
            return self.raw_item.model_dump(exclude_unset=True)
        return self.raw_item


@dataclass
class MessageOutputItem(RunItemBase[ResponseOutputItem]):
    """LLM message output."""
    raw_item: ResponseOutputItem
    type: Literal["message_output_item"] = "message_output_item"


@dataclass
class HandoffCallItem(RunItemBase[ResponseFunctionToolCall]):
    """Agent handoff tool call."""
    raw_item: ResponseFunctionToolCall
    type: Literal["handoff_call_item"] = "handoff_call_item"


@dataclass
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


@dataclass
class ToolCallItem(RunItemBase[ToolCallItemTypes]):
    """Tool call for function or computer action."""
    raw_item: ToolCallItemTypes
    type: Literal["tool_call_item"] = "tool_call_item"


@dataclass
class ToolCallOutputItem(RunItemBase[Union[FunctionCallOutput, ComputerCallOutput]]):
    """Tool call execution output."""
    raw_item: FunctionCallOutput | ComputerCallOutput
    output: Any
    type: Literal["tool_call_output_item"] = "tool_call_output_item"


@dataclass
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


@dataclass
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

    def to_input_items(self) -> list[TResponseInputItem]:
        """Convert outputs to input format.

        Returns:
            List of items converted to input format
        """
        return [it.model_dump(exclude_unset=True) for it in self.output]  # type: ignore


########################################################
#                   Item Helpers                        #
########################################################

class ItemHelpers:
    """Helper methods for processing and formatting various item types."""

    @classmethod
    def extract_last_content(cls, message: TResponseOutputItem) -> str:
        """Extract the last content from a message.

        Args:
            message: The message to extract content from

        Returns:
            The last text or refusal content

        Raises:
            ModelError: If the content type is unexpected
        """
        if not hasattr(message, "type") or message.type != "message":
            return ""

        last_content = message.content[-1]
        if last_content.type == "output_text":
            return last_content.text
        elif last_content.type == "refusal":
            return last_content.refusal
        else:
            raise ModelError(f"Unexpected content type: {last_content.type}")

    @classmethod
    def extract_last_text(cls, message: TResponseOutputItem) -> str | None:
        """Extract the last text content from a message, ignoring refusals.

        Args:
            message: The message to extract text from

        Returns:
            The last text content if available, None otherwise
        """
        if hasattr(message, "type") and message.type == "message":
            last_content = message.content[-1]
            if last_content.type == "output_text":
                return last_content.text
        return None

    @classmethod
    def input_to_new_input_list(
        cls, input: str | list[TResponseInputItem]
    ) -> list[TResponseInputItem]:
        """Convert string or input items to input list.

        Args:
            input: String or list of input items to convert

        Returns:
            List of input items
        """
        if isinstance(input, str):
            return [{"content": input, "role": "user"}]
        return copy.deepcopy(input)

    @classmethod
    def text_message_outputs(cls, items: list[RunItem]) -> str:
        """Concatenate text from all message outputs.

        Args:
            items: List of run items to process

        Returns:
            Concatenated text from message outputs
        """
        return "".join(
            cls.text_message_output(item)
            for item in items
            if isinstance(item, MessageOutputItem)
        )

    @classmethod
    def text_message_output(cls, message: MessageOutputItem) -> str:
        """Extract text from a message output.

        Args:
            message: The message output to extract text from

        Returns:
            Extracted text content
        """
        return "".join(
            item["text"]
            for item in message.raw_item["content"]
            if item["type"] == "output_text"
        )

    @classmethod
    def tool_call_output_item(
        cls, tool_call: ResponseFunctionToolCall, output: str
    ) -> FunctionCallOutput:
        """Create a tool call output from a call and result.

        Args:
            tool_call: The tool call to create output for
            output: The output string

        Returns:
            Function call output dictionary
        """
        return {
            "call_id": tool_call.call_id,
            "output": output,
            "type": "function_call_output",
        }

    @classmethod
    def format_content(cls, content: str) -> str:
        """Format content with proper indentation and borders.

        Args:
            content: The content string to format

        Returns:
            Formatted content with proper indentation and borders
        """
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        formatted_lines = []
        current_section = []

        def format_section(section: list[str]) -> list[str]:
            if not section:
                return []
            width = max(len(l) for l in section)
            border = "+" + "-" * (width + 2) + "+"
            return [
                border,
                *[f"| {l:<{width}} |" for l in section],
                border
            ]

        for line in lines:
            if line.startswith('<think>') or line.startswith('Echoes of encrypted hearts'):
                if current_section:
                    formatted_lines.extend(format_section(current_section))
                    current_section = []
                if line.startswith('Echoes of encrypted hearts'):
                    formatted_lines.append('')
                formatted_lines.append(line)
            elif line.endswith('</think>'):
                formatted_lines.append(line)
            else:
                current_section.append(line)

        if current_section:
            formatted_lines.extend(format_section(current_section))

        return '\n'.join(formatted_lines)
