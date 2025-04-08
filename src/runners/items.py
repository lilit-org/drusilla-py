"""
This module provides core data structures and utilities for handling different
types of items and responses in an AI agent system. It defines various item
types including messages, function calls (sword/orbs calls), reasoning items,
and their corresponding outputs.

Key Components:
- RunItemBase: Abstract base class for all run items with common functionality
- MessageOutputItem: Handles text message outputs and content extraction
- SwordCallItem/OrbsCallItem: Manage function call requests
- SwordCallOutputItem/OrbsOutputItem: Handle function call responses
- ReasoningItem: Processes reasoning steps and thought processes
- ItemHelpers: Utility class with helper methods for item manipulation
"""

from __future__ import annotations

import abc
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

from ..util.types import (
    FunctionCallOutput,
    ResponseFunctionSwordCall,
    ResponseInputItemParam,
    ResponseReasoningItem,
    Usage,
)
from ..util.types import (
    ResponseOutput as TResponseOutputItem,
)

if TYPE_CHECKING:
    from ..agents.agent_v1 import Agent


########################################################
#            Type Aliases and Constants                #
########################################################

T = TypeVar("T")

MESSAGE_TYPE = "message"
OUTPUT_TEXT_TYPE = "output_text"
REFUSAL_TYPE = "refusal"
THINK_START = "<think>"
THINK_END = "</think>"

RunItem = Union[
    "MessageOutputItem",
    "OrbsCallItem",
    "OrbsOutputItem",
    "SwordCallItem",
    "SwordCallOutputItem",
    "ReasoningItem",
]

########################################################
#                        Data Classes
########################################################


@dataclass(frozen=True)
class RunItemBase(Generic[T], abc.ABC):
    agent: Agent[Any]
    raw_item: T

    @cached_property
    def input_item(self) -> ResponseInputItemParam:
        if isinstance(self.raw_item, dict):
            return self.raw_item
        elif isinstance(self.raw_item, BaseModel):
            return self.raw_item.model_dump(exclude_unset=True)
        return self.raw_item


@dataclass(frozen=True)
class MessageOutputItem(RunItemBase[TResponseOutputItem]):
    raw_item: TResponseOutputItem
    type: Literal["message_output_item"] = "message_output_item"

    @cached_property
    def text_content(self) -> str:
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
                        text = item.get("text", "").strip()
                        if text:
                            texts.append(text)
                elif hasattr(item, "type") and item.type == OUTPUT_TEXT_TYPE:
                    text = getattr(item, "text", "").strip()
                    if text:
                        texts.append(text)

            return " ".join(texts)

        except (AttributeError, KeyError, TypeError, IndexError) as e:
            print(f"Error getting text content: {e}")
            return ""


@dataclass(frozen=True)
class OrbsCallItem(RunItemBase[ResponseFunctionSwordCall]):
    raw_item: ResponseFunctionSwordCall
    type: Literal["orbs_call_item"] = "orbs_call_item"


@dataclass(frozen=True)
class OrbsOutputItem(RunItemBase[ResponseInputItemParam]):
    raw_item: ResponseInputItemParam
    source_agent: Agent[Any]
    target_agent: Agent[Any]
    type: Literal["orbs_output_item"] = "orbs_output_item"


@dataclass(frozen=True)
class SwordCallItem(RunItemBase[ResponseFunctionSwordCall]):
    raw_item: ResponseFunctionSwordCall
    type: Literal["sword_called"] = "sword_called"


@dataclass(frozen=True)
class SwordCallOutputItem(RunItemBase[FunctionCallOutput]):
    raw_item: FunctionCallOutput
    output: Any
    type: Literal["sword_call_output_item"] = "sword_call_output_item"


@dataclass(frozen=True)
class ReasoningItem(RunItemBase[ResponseReasoningItem]):
    raw_item: ResponseReasoningItem
    type: Literal["reasoning_item"] = "reasoning_item"


@dataclass(frozen=True)
class ModelResponse:
    output: list[TResponseOutputItem]
    usage: Usage
    referenceable_id: str | None

    @cached_property
    def input_items(self) -> list[ResponseInputItemParam]:
        return [
            cast(
                ResponseInputItemParam,
                it.model_dump(exclude_unset=True) if hasattr(it, "model_dump") else it,
            )
            for it in self.output
        ]

    def to_input_items(self) -> list[ResponseInputItemParam]:
        return self.input_items


class ItemHelpers:
    SPECIAL_LINES: ClassVar[tuple[str]] = (THINK_START, THINK_END)

    @staticmethod
    def extract_last_content(message: TResponseOutputItem) -> str:
        try:
            message_type = (
                message.get("type") if isinstance(message, dict) else getattr(message, "type", None)
            )
            if message_type != MESSAGE_TYPE:
                return ""

            content = (
                message.get("content", [])
                if isinstance(message, dict)
                else getattr(message, "content", [])
            )
            if not content:
                return ""

            last_content = content[-1]
            content_type = (
                last_content.get("type")
                if isinstance(last_content, dict)
                else getattr(last_content, "type", None)
            )

            if content_type == OUTPUT_TEXT_TYPE:
                return (
                    last_content.get("text", "")
                    if isinstance(last_content, dict)
                    else getattr(last_content, "text", "")
                )
            if content_type == REFUSAL_TYPE:
                return (
                    last_content.get("refusal", "")
                    if isinstance(last_content, dict)
                    else getattr(last_content, "refusal", "")
                )
            return ""
        except (AttributeError, IndexError, KeyError):
            return ""

    @staticmethod
    def extract_last_text(message: TResponseOutputItem) -> str | None:
        try:
            message_type = (
                message.get("type") if isinstance(message, dict) else getattr(message, "type", None)
            )
            if message_type != MESSAGE_TYPE:
                return None

            content = (
                message.get("content", [])
                if isinstance(message, dict)
                else getattr(message, "content", [])
            )
            if not content:
                return None

            last_content = content[-1]
            content_type = (
                last_content.get("type")
                if isinstance(last_content, dict)
                else getattr(last_content, "type", None)
            )

            if content_type == OUTPUT_TEXT_TYPE:
                return (
                    last_content.get("text", "")
                    if isinstance(last_content, dict)
                    else getattr(last_content, "text", "")
                )
            return None
        except (AttributeError, IndexError, KeyError):
            return None

    @staticmethod
    def input_to_new_input_list(
        input: str | list[ResponseInputItemParam],
    ) -> list[ResponseInputItemParam]:
        if isinstance(input, str):
            return [{"content": input, "role": "user"}]
        return input.copy()

    @staticmethod
    def text_message_outputs(items: list[RunItem]) -> str:
        return " ".join(
            item.text_content if isinstance(item, MessageOutputItem) else "" for item in items
        ).strip()

    @staticmethod
    def text_message_output(message: MessageOutputItem) -> str:
        try:
            if not hasattr(message, "raw_item"):
                return ""

            raw_item = message.raw_item
            if not isinstance(raw_item, dict):
                return ""

            content = raw_item.get("content", [])
            if not content:
                return ""

            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == OUTPUT_TEXT_TYPE:
                    text = item.get("text", "").strip()
                    if text:
                        text_parts.append(text)

            if not text_parts:
                return ""

            text = " ".join(text_parts)
            text = text.replace("', 'type': 'output_text', 'annotations': []}", "")
            text = text.replace("'text': '", "").replace("'", "").strip()

            if not text:
                return ""

            lines = []
            current_section = []

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if line.startswith(THINK_START):
                    if current_section:
                        lines.append(" ".join(current_section))
                        current_section = []
                    lines.append(line)
                elif line.endswith(THINK_END):
                    if current_section:
                        lines.append(" ".join(current_section))
                        current_section = []
                    lines.append(line)
                else:
                    current_section.append(line)

            if current_section:
                lines.append(" ".join(current_section))

            return "\n".join(lines)

        except Exception as e:
            print(f"Error processing message output: {e}")
            return ""

    @staticmethod
    def sword_call_output_item(
        sword_call: ResponseFunctionSwordCall, output: str
    ) -> FunctionCallOutput:
        return {
            "call_id": sword_call["call_id"],
            "output": output,
            "type": "function_call_output",
        }

    @staticmethod
    def format_content(content: str) -> str:
        if not content:
            return ""
        content = content.replace("', 'type': 'output_text', 'annotations': []}", "")
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if not lines:
            return ""

        # If there's only one think section, return just the content
        if len(lines) == 1 and lines[0].startswith(THINK_START) and lines[0].endswith(THINK_END):
            return lines[0][len(THINK_START) : -len(THINK_END)]

        # If there are no special lines, just return the content as is
        if not any(line.startswith(ItemHelpers.SPECIAL_LINES) for line in lines):
            return " ".join(lines)

        # For multiple think sections, extract content between think tags
        result = []
        for line in lines:
            if line.startswith(THINK_START) and line.endswith(THINK_END):
                result.append(line[len(THINK_START) : -len(THINK_END)])
            elif not line.startswith(ItemHelpers.SPECIAL_LINES):
                result.append(line)

        return "\n".join(result)


__all__ = [
    "ItemHelpers",
    "MessageOutputItem",
    "ModelResponse",
    "OrbsCallItem",
    "OrbsOutputItem",
    "SwordCallItem",
    "SwordCallOutputItem",
    "ReasoningItem",
    "THINK_START",
    "THINK_END",
]
