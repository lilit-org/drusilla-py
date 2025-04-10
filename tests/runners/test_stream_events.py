from dataclasses import dataclass
from typing import Any

import pytest

from src.runners.items import RunItemBase
from src.runners.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    RunItemStreamEventName,
    StreamEvent,
)
from src.util.types import ResponseStreamEvent
from src.util.exceptions import ModelError


# Mock classes for testing
@dataclass
class MockAgent:
    name: str


@dataclass(frozen=True)
class MockRunItem(RunItemBase[Any]):
    type: str = "test_item"
    raw_item: Any = None
    agent: Any = None


def test_raw_responses_stream_event():
    """Test RawResponsesStreamEvent creation and attributes."""
    response_event = ResponseStreamEvent(
        type="test", content_index=0, item_id="test_id", output_index=0
    )

    event = RawResponsesStreamEvent(data=response_event)

    assert event.type == "raw_response_event"
    assert event.data == response_event
    assert event.data["type"] == "test"
    assert event.data["content_index"] == 0
    assert event.data["item_id"] == "test_id"
    assert event.data["output_index"] == 0


def test_run_item_stream_event():
    """Test RunItemStreamEvent creation and attributes."""
    item = MockRunItem()
    event = RunItemStreamEvent(name="message_output_created", item=item)

    assert event.type == "run_item_stream_event"
    assert event.name == "message_output_created"
    assert event.item == item
    assert event.item.type == "test_item"


def test_agent_updated_stream_event():
    """Test AgentUpdatedStreamEvent creation and attributes."""
    agent = MockAgent(name="test_agent")
    event = AgentUpdatedStreamEvent(new_agent=agent)

    assert event.type == "agent_updated_stream_event"
    assert event.new_agent == agent
    assert event.new_agent.name == "test_agent"


def test_stream_event_type_alias():
    """Test StreamEvent type alias usage."""
    # Test RawResponsesStreamEvent
    response_event = ResponseStreamEvent(type="test")
    raw_event = RawResponsesStreamEvent(data=response_event)
    assert isinstance(raw_event, StreamEvent)

    # Test RunItemStreamEvent
    item = MockRunItem()
    run_event = RunItemStreamEvent(name="message_output_created", item=item)
    assert isinstance(run_event, StreamEvent)

    # Test AgentUpdatedStreamEvent
    agent = MockAgent(name="test_agent")
    agent_event = AgentUpdatedStreamEvent(new_agent=agent)
    assert isinstance(agent_event, StreamEvent)


def test_run_item_stream_event_name():
    """Test RunItemStreamEventName type validation."""
    valid_names: list[RunItemStreamEventName] = [
        "message_output_created",
        "orbs_requested",
        "orbs_occurred",
        "sword_called",
        "sword_output",
        "reasoning_item_created",
    ]

    for name in valid_names:
        event = RunItemStreamEvent(name=name, item=MockRunItem())
        assert event.name == name

    # Test invalid name (should raise model error)
    with pytest.raises(ModelError):
        RunItemStreamEvent(name="invalid_name", item=MockRunItem())  # type: ignore
