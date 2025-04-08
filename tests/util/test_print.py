from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter

from src.runners.items import ModelResponse, Usage
from src.util.exceptions import ModelError
from src.util.print import (
    _format_result,
    _format_stats,
    _format_stream_info,
    _indent,
    pretty_print_result,
    pretty_print_result_stats,
    validate_json,
)
from src.util.types import ResponseOutput


# Mock classes for testing
@dataclass
class MockAgent:
    name: str
    swords: list[Any] = None


@dataclass
class MockRunResult:
    current_agent: MockAgent
    current_turn: int
    max_turns: int
    is_complete: bool
    new_items: list[Any]
    raw_responses: list[Any]
    input_shield_results: list[Any]
    output_shield_results: list[Any]
    last_agent: MockAgent


@dataclass
class MockRunResultStreaming:
    text: str
    current_agent: MockAgent
    current_turn: int
    max_turns: int
    is_complete: bool
    last_agent: MockAgent


def test_indent():
    """Test _indent function."""
    text = "line1\nline2\nline3"
    indented = _indent(text, 2)
    assert indented == "    line1\n    line2\n    line3"

    # Test empty string
    assert _indent("", 2) == ""

    # Test single line
    assert _indent("single line", 1) == "  single line"


def test_format_result():
    """Test _format_result function."""
    # Test case 1: With reasoning
    response = ModelResponse(
        output=[ResponseOutput(type="text", text="<think>Reasoning</think>Result")],
        usage=Usage(),
        referenceable_id=None,
    )
    result = _format_result(response, show_reasoning=True)
    assert "Reasoning" in result
    assert "Result" in result

    # Test case 2: Without reasoning
    result = _format_result(response, show_reasoning=False)
    assert "Reasoning" not in result
    assert "Result" in result

    # Test case 3: Empty response
    empty_response = ModelResponse(output=[], usage=Usage(), referenceable_id=None)
    result = _format_result(empty_response)
    assert result == ""


def test_format_stats():
    """Test _format_stats function."""
    result = MockRunResult(
        current_agent=MockAgent(name="test"),
        current_turn=1,
        max_turns=3,
        is_complete=True,
        new_items=[1, 2, 3],
        raw_responses=[1, 2],
        input_shield_results=[1],
        output_shield_results=[1, 2],
        last_agent=MockAgent(name="test"),
    )
    stats = _format_stats(result)
    assert "3" in stats  # Items
    assert "2" in stats  # Responses
    assert "1" in stats  # Input Shield
    assert "2" in stats  # Output Shield

    # Test case 2: Empty lists
    result.new_items = []
    result.raw_responses = []
    result.input_shield_results = []
    result.output_shield_results = []
    stats = _format_stats(result)
    assert "0" in stats  # All counts should be 0


def test_format_stream_info():
    """Test _format_stream_info function."""
    agent = MockAgent(name="test", swords=["sword1", "sword2"])
    result = MockRunResult(
        current_agent=agent,
        current_turn=1,
        max_turns=3,
        is_complete=True,
        new_items=[],
        raw_responses=[],
        input_shield_results=[],
        output_shield_results=[],
        last_agent=agent,
    )

    # Test case 1: Streaming enabled
    info = _format_stream_info(True, result)
    assert "Enabled" in info
    assert "2 swords" in info

    # Test case 2: Streaming disabled
    info = _format_stream_info(False, result)
    assert "Disabled" in info

    # Test case 3: No swords
    agent.swords = []
    info = _format_stream_info(True, result)
    assert "None" in info

    # Test case 4: No agent
    info = _format_stream_info(True, None)
    assert "Enabled" in info
    assert "None" not in info


def test_pretty_print_result_stats():
    """Test pretty_print_result_stats function."""
    agent = MockAgent(name="test", swords=["sword1"])
    result = MockRunResult(
        current_agent=agent,
        current_turn=1,
        max_turns=3,
        is_complete=True,
        new_items=[1],
        raw_responses=[1],
        input_shield_results=[1],
        output_shield_results=[1],
        last_agent=agent,
    )

    stats = pretty_print_result_stats(result)
    assert "Performance" in stats
    assert "test" in stats
    assert "1/3" in stats
    assert "Complete" in stats


def test_pretty_print_result():
    """Test pretty_print_result function."""
    response = ModelResponse(
        output=[ResponseOutput(type="text", text="<think>Reasoning</think>Result")],
        usage=Usage(),
        referenceable_id=None,
    )
    result = MockRunResult(
        current_agent=MockAgent(name="test"),
        current_turn=1,
        max_turns=3,
        is_complete=True,
        new_items=[],
        raw_responses=[response],
        input_shield_results=[],
        output_shield_results=[],
        last_agent=MockAgent(name="test"),
    )

    # Test case 1: Normal case
    output = pretty_print_result(result)
    assert "Reasoning" in output
    assert "Result" in output

    # Test case 2: No raw responses
    result.raw_responses = []
    with pytest.raises(ModelError) as exc_info:
        pretty_print_result(result)
    assert "No raw responses found in result" in str(exc_info.value)

    # Test case 3: With reasoning disabled
    result.raw_responses = [response]
    output = pretty_print_result(result, show_reasoning=False)
    assert "Reasoning" not in output
    assert "Result" in output


def test_validate_json():
    """Test validate_json function."""

    class TestModel(BaseModel):
        name: str
        age: int

    adapter = TypeAdapter(TestModel)

    # Test case 1: Valid JSON
    json_str = '{"name": "John", "age": 30}'
    result = validate_json(json_str, adapter)
    assert result.name == "John"
    assert result.age == 30

    # Test case 2: Invalid JSON
    json_str = '{"name": "John", "age": "thirty"}'
    with pytest.raises(ModelError) as exc_info:
        validate_json(json_str, adapter)
    assert "validation error" in str(exc_info.value).lower()

    # Test case 3: Partial validation
    json_str = '{"name": "John"}'
    with pytest.raises(ModelError) as exc_info:
        validate_json(json_str, adapter, partial=False)
    assert "validation error" in str(exc_info.value).lower()

    # Test case 4: Partial validation allowed
    class PartialTestModel(BaseModel):
        name: str
        age: int | None = None

    adapter = TypeAdapter(PartialTestModel)
    result = validate_json(json_str, adapter, partial=True)
    assert result.name == "John"
    assert result.age is None
