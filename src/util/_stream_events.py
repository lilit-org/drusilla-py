from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

from typing_extensions import TypeAlias

from ..agents.agent import Agent
from ._items import RunItem, TResponseStreamEvent

########################################################
#               Data class RawResponsesStreamEvent      #
########################################################

@dataclass
class RawResponsesStreamEvent:
    """Direct pass-through events from the LLM."""

    data: TResponseStreamEvent
    """Raw LLM response event."""

    type: Literal["raw_response_event"] = "raw_response_event"
    """Event type."""


########################################################
#               Data class RunItemStreamEvent          #
########################################################

@dataclass
class RunItemStreamEvent:
    """Events generated during agent processing of LLM responses."""

    name: Literal[
        "message_output_created",
        "handoff_requested",
        "handoff_occured",
        "tool_called",
        "tool_output",
        "reasoning_item_created",
    ]
    """Event name."""

    item: RunItem
    """Created item."""

    type: Literal["run_item_stream_event"] = "run_item_stream_event"


########################################################
#               Data class AgentUpdatedStreamEvent     #
########################################################

@dataclass
class AgentUpdatedStreamEvent:
    """Notification of a new agent instance."""

    new_agent: Agent[Any]
    """New agent instance."""

    type: Literal["agent_updated_stream_event"] = "agent_updated_stream_event"


########################################################
#              StreamEvent Type Alias                  #
########################################################

StreamEvent: TypeAlias = Union[RawResponsesStreamEvent, RunItemStreamEvent, AgentUpdatedStreamEvent]
"""Agent streaming event type."""
