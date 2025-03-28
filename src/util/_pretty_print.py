import re
from typing import Any

from ._items import ModelResponse
from ._result import RunResult

########################################################
#               Constants
########################################################

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)

########################################################
#          Final Output private method
########################################################


def _indent(text: str, indent_level: int) -> str:
    return "\n".join("  " * indent_level + line for line in text.splitlines())


def _format_final_output(raw_response: ModelResponse) -> str:
    output = raw_response.output[0]["text"]
    match = _THINK_PATTERN.search(output)

    if match:
        reasoning = match.group(1).strip().encode().decode("unicode-escape")
        final_result = match.group(2).strip().encode().decode("unicode-escape")
    else:
        reasoning = ""
        final_result = output.strip("'").strip().encode().decode("unicode-escape")

    return f"\n\n✅ REASONING:\n\n{reasoning}\n\n✅ RESULT:\n\n{final_result}\n"


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


def pretty_print_result(result: RunResult) -> str:
    parts = [
        f"✅ {result.__class__.__name__}:",
        _format_agent_info(result),
        _format_stats(result),
        _format_stream_info(
            stream=hasattr(result, "is_complete"),
            tool_choice=getattr(result, "tool_choice", None),
            result=result,
        ),
        _format_final_output(result.raw_responses[0]),
    ]
    return "".join(parts)
