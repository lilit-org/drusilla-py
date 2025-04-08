"""
This module provides utilities for formatting and printing the results of agent runs.
It includes:

1. Pretty-printing functions for:
   - Basic result summaries
   - Detailed statistics and configuration
   - Agent status and turn information
   - Streaming event handling

2. JSON validation and transformation utilities
   - Validates JSON strings against Pydantic models
   - Converts strings into valid Python function names
"""

import re
from typing import Any

from pydantic import TypeAdapter, ValidationError

from src.util.constants import ERROR_MESSAGES, THINK_TAGS

from ..runners.items import ModelResponse, Usage
from ..runners.result import RunResult, RunResultStreaming
from .exceptions import ModelError
from .types import T

########################################################
#              Constants
########################################################

FUNCTION_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9]")

########################################################
#         Private method
########################################################


def _indent(text: str, indent_level: int) -> str:
    """Indent each line of text by the specified number of spaces."""
    return "\n".join("  " * indent_level + line for line in text.splitlines())


def _format_final_output(raw_response: ModelResponse) -> tuple[str, str]:
    """Format the model response into reasoning and result components."""
    if not raw_response.output:
        return "", ""

    try:
        if not isinstance(raw_response.output, list):
            raise ModelError(
                ERROR_MESSAGES.MODEL_ERROR.message.format(error="Output is not a list")
            )

        if not raw_response.output:
            return "", ""

        output_text = str(raw_response.output[0].get("text", raw_response.output[0]))
        output_text = output_text.strip("'").strip().encode().decode("unicode-escape")

        if THINK_TAGS[0] not in output_text or THINK_TAGS[1] not in output_text:
            return "", f"\n\n✅ RESULT:\n\n{output_text}\n"

        start = output_text.find(THINK_TAGS[0]) + len(THINK_TAGS[0])
        end = output_text.find(THINK_TAGS[1])
        reasoning = output_text[start:end].strip()
        result = output_text[end + len(THINK_TAGS[1]) :].strip()
        return (f"\n\n✅ REASONING:\n\n{reasoning}", f"\n\n✅ RESULT:\n\n{result}\n")
    except (IndexError, AttributeError, UnicodeDecodeError) as e:
        raise ModelError(
            ERROR_MESSAGES.MODEL_ERROR.message.format(
                error=f"Failed to format model output: {str(e)}"
            )
        ) from e


def _format_result(result: Any, show_reasoning: bool = True) -> str:
    """Format the result with optional reasoning display."""
    res_reasoning, res_result = _format_final_output(result)
    return (res_reasoning + res_result) if show_reasoning else res_result


def _format_agent_info(result: Any) -> str:
    """Format agent information into a readable string."""
    info = ["\n👾 Agent Info:"]

    if isinstance(result, RunResultStreaming):
        info.extend(
            [
                f"      Name   → {result.current_agent.name}",
                f"      Turn   → {result.current_turn}/{result.max_turns}",
                f"      Status → {'✔️ Complete' if result.is_complete else '🟡 Running'}",
            ]
        )
    else:
        info.append(f"      Last Agent → {result.last_agent.name}")
        if hasattr(result, "current_turn") and hasattr(result, "max_turns"):
            info.append(f"      Turn       → {result.current_turn}/{result.max_turns}")
        info.append(f"      Status     → {'✔️ Complete' if result.is_complete else '🟡 Running'}")

    return "\n" + "\n".join(_indent(line, 1) for line in info)


def _format_stats(result: Any) -> str:
    """Format statistics into a readable string."""
    stats = [
        "\n📊 Statistics:",
        f"      Items     → {len(result.new_items)}",
        f"      Responses → {len(result.raw_responses)}",
        f"      Input Shield  → {len(result.input_shield_results)}",
        f"      Output Shield → {len(result.output_shield_results)}",
    ]
    return "\n" + "\n".join(_indent(stat, 1) for stat in stats)


def _format_stream_info(stream: bool, result: Any) -> str:
    """Format streaming information into a readable string."""

    def format_obj(x: Any) -> str:
        if x is None or x is object():
            return "None"
        if isinstance(x, bool):
            return "✔️ Enabled" if x else "❌ Disabled"
        if isinstance(x, list):
            return f"✔️ Available ({len(x)} swords)" if x else "None"
        return str(x)

    info = ["\n🦾 Configuration:"]
    swords = getattr(result, "last_agent", None)
    info.append(f"      Streaming → {format_obj(stream)}")
    if swords and hasattr(swords, "swords"):
        info.append(f"      Swords    → {format_obj(swords.swords)}")
    return "\n" + "\n".join(_indent(line, 1) for line in info)


########################################################
#               Public methods
########################################################


def pretty_print_result_stats(result: RunResult) -> str:
    parts = [
        f"\n✅ {result.__class__.__name__} Performance:",
        _format_agent_info(result),
        _format_stats(result),
        _format_stream_info(
            stream=hasattr(result, "is_complete"),
            result=result,
        ),
    ]
    return "".join(parts)


def pretty_print_result(result: RunResult, show_reasoning: bool = True) -> str:
    """Format and print the result of a run with optional reasoning display."""
    if not result.raw_responses:
        raise ModelError(
            ERROR_MESSAGES.MODEL_ERROR.message.format(error="No raw responses found in result")
        )
    try:
        return _format_result(result.raw_responses[0], show_reasoning)
    except (IndexError, AttributeError) as e:
        raise ModelError(
            ERROR_MESSAGES.MODEL_ERROR.message.format(error=f"Invalid result format: {str(e)}")
        ) from e


def pretty_print_result_stream(result: RunResultStreaming, show_reasoning: bool = True) -> str:
    return _format_result(
        ModelResponse(output=[{"text": result}], usage=Usage(), referenceable_id=None),
        show_reasoning,
    )


def validate_json(json_str: str, type_adapter: TypeAdapter[T], partial: bool = False) -> T:
    """Validates a JSON string against a type adapter. Raises ModelError if invalid."""
    try:
        return type_adapter.validate_json(json_str, experimental_allow_partial=partial)
    except ValidationError as e:
        raise ModelError(ERROR_MESSAGES.MODEL_ERROR.message.format(error=str(e))) from e


def transform_string_function_style(name: str) -> str:
    """Converts a string into a valid Python function name."""
    return FUNCTION_NAME_PATTERN.sub("_", name.replace(" ", "_")).lower()
