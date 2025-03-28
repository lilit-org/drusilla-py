import re
from typing import Any

from ._items import ModelResponse, Usage
from ._result import RunResult, RunResultStreaming

########################################################
#          Final Output private method
########################################################


def _indent(text: str, indent_level: int) -> str:
    return "\n".join("  " * indent_level + line for line in text.splitlines())


def _format_final_output(raw_response: ModelResponse) -> str:
    if not raw_response.output:
        return ""

    # Get text from the first output item
    output_text = str(raw_response.output[0].get("text", raw_response.output[0]))
    output_text = output_text.strip("'").strip().encode().decode("unicode-escape")

    # Extract everything between think tags for reasoning
    reasoning = ""
    result = output_text

    if "<think>" in output_text and "</think>" in output_text:
        # Find the start and end positions
        start = output_text.find("<think>") + len("<think>")
        end = output_text.find("</think>")

        # Extract the reasoning (everything between think tags)
        reasoning = output_text[start:end].strip()

        # Extract the result (everything after </think>)
        result = output_text[end + len("</think>") :].strip()

        # Clean up any leading/trailing whitespace or newlines
        reasoning = reasoning.strip()
        result = result.strip()

        # Remove "Here are the jokes:" if present
        result = re.sub(
            r"^Here are the jokes?:", "", result, flags=re.IGNORECASE
        ).strip()

    return f"\n\n✅ REASONING:\n\n{reasoning}\n\n✅ RESULT:\n\n{result}\n"


########################################################
#               Agent Info private method
########################################################


def _format_agent_info(result: Any) -> str:
    info: list[str] = ["\n👾 Agent Info:"]
    if hasattr(result, "is_complete"):
        info.extend(
            [
                f"      Name   → {result.current_agent.name}",
                f"      Turn   → {result.current_turn}/{result.max_turns}",
                f"      Status → {'✔️ Complete' if result.is_complete else '🟡 Running'}",
            ]
        )
    else:
        info.append(f"      Last Agent → {result.last_agent.name}")
    return "\n" + "\n".join(_indent(line, 1) for line in info)


########################################################
#               Stats Section private method
########################################################


def _format_stats(result: Any) -> str:
    stats = [
        "\n📊 Statistics:",
        f"      Items     → {len(result.new_items)}",
        f"      Responses → {len(result.raw_responses)}",
        f"      Input GR  → {len(result.input_shield_results)}",
        f"      Output GR → {len(result.output_shield_results)}",
    ]
    return "\n" + "\n".join(_indent(stat, 1) for stat in stats)


########################################################
#               Stream Info private method
########################################################


def _format_stream_info(stream: bool, tool_choice: Any, result: Any) -> str:
    def format_obj(x: Any) -> str:
        if x is None or x is object():
            return "None"
        if isinstance(x, bool):
            return "✔️ Enabled" if x else "❌ Disabled"
        if isinstance(x, list):
            return f"Available ({len(x)} tools)" if x else "None"
        return str(x)

    info = ["\n🦾 Configuration:"]
    tools = getattr(result, "last_agent", None)
    info.append(f"      Streaming → {format_obj(stream)}")
    if tools and hasattr(tools, "tools"):
        info.append(f"      Tools     → {format_obj(tools.tools)}")
    info.append(f"      Tool Mode → {format_obj(tool_choice)}")
    return "\n" + "\n".join(_indent(line, 1) for line in info)


########################################################
#               Public Main method
########################################################


def pretty_print_result_stats(result: RunResult) -> str:
    parts = [
        f"\n✅ {result.__class__.__name__} Performance:",
        _format_agent_info(result),
        _format_stats(result),
        _format_stream_info(
            stream=hasattr(result, "is_complete"),
            tool_choice=getattr(result, "tool_choice", None),
            result=result,
        ),
    ]
    return "".join(parts)


def pretty_print_result(result: RunResult) -> str:
    return pretty_print_result_stats(result) + _format_final_output(
        result.raw_responses[0]
    )


def pretty_print_result_stream(result: RunResultStreaming):
    return _format_final_output(
        ModelResponse(output=[{"text": result}], usage=Usage(), referenceable_id=None)
    )
