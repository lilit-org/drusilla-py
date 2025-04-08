"""
This module defines a comprehensive event system for handling streaming data in the application.
It provides three main types of stream events:

1. RawResponsesStreamEvent: Captures direct responses from the Language Model (LLM)
2. RunItemStreamEvent: Tracks various stages of agent processing including:
   - Message outputs
   - Orb requests and occurrences
   - Sword operations
   - Reasoning item creation
3. AgentUpdatedStreamEvent: Notifies when agent instances are updated

These events form the backbone of the streaming data pipeline, enabling real-time processing
and state management throughout the application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from ..util.types import ResponseStreamEvent
from .items import RunItem

if TYPE_CHECKING:
    from ..agents.agent_v1 import Agent

########################################################
#               Type Aliases                           #
########################################################

RunItemStreamEventName: TypeAlias = Literal[
    "message_output_created",
    "orbs_requested",
    "orbs_occurred",
    "sword_called",
    "sword_output",
    "reasoning_item_created",
]

########################################################
#               Data class RawResponsesStreamEvent      #
########################################################


@dataclass(frozen=True)
class RawResponsesStreamEvent:
    """Direct pass-through events from the LLM."""

    data: ResponseStreamEvent
    type: Literal["raw_response_event"] = "raw_response_event"


########################################################
#               Data class RunItemStreamEvent          #
########################################################

VALID_RUN_ITEM_EVENT_NAMES = {
    "message_output_created",
    "orbs_requested",
    "orbs_occurred",
    "sword_called",
    "sword_output",
    "reasoning_item_created",
}


@dataclass(frozen=True)
class RunItemStreamEvent:
    """Events generated during agent processing of LLM responses."""

    name: RunItemStreamEventName
    item: RunItem
    type: Literal["run_item_stream_event"] = "run_item_stream_event"

    def __post_init__(self):
        if self.name not in VALID_RUN_ITEM_EVENT_NAMES:
            raise TypeError(f"Invalid RunItemStreamEventName: {self.name}")


########################################################
#               Data class AgentUpdatedStreamEvent     #
########################################################


@dataclass(frozen=True)
class AgentUpdatedStreamEvent:
    """Notification of a new agent instance."""

    new_agent: Agent[Any]
    type: Literal["agent_updated_stream_event"] = "agent_updated_stream_event"


########################################################
#              StreamEvent Type Alias                  #
########################################################

StreamEvent: TypeAlias = RawResponsesStreamEvent | RunItemStreamEvent | AgentUpdatedStreamEvent
