from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

from typing_extensions import TypeAlias

from ..agents.agent import Agent
from ._items import RunItem, TResponseStreamEvent

########################################################
#               Data class RawResponsesStreamEvent      #
########################################################

@dataclass(frozen=True)
class RawResponsesStreamEvent:
    """Direct pass-through events from the LLM."""
    data: TResponseStreamEvent
    type: Literal["raw_response_event"] = "raw_response_event"



########################################################
#               Data class RunItemStreamEvent          #
########################################################

@dataclass(frozen=True)
class RunItemStreamEvent:
    """Events generated during agent processing of LLM responses."""

    name: Literal[
        "message_output_created",
        "handoff_requested",
        "handoff_occurred",
        "tool_called",
        "tool_output",
        "reasoning_item_created",
    ]
    item: RunItem
    type: Literal["run_item_stream_event"] = "run_item_stream_event"


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

StreamEvent: TypeAlias = Union[RawResponsesStreamEvent, RunItemStreamEvent, AgentUpdatedStreamEvent]
