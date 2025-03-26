import re
from typing import Any, Protocol

from pydantic import BaseModel

########################################################
#               Protocol Class                         #
########################################################

class PrettyPrintable(Protocol):
    last_agent: Any
    current_agent: Any
    current_turn: int
    max_turns: int
    is_complete: bool
    final_output: Any
    new_items: list
    raw_responses: list
    input_guardrail_results: list
    output_guardrail_results: list


########################################################
#               Private Functions                      #
########################################################

def _indent(text: str, indent_level: int) -> str:
    """Indent each line of text by the specified number of spaces."""
    indent_string = "  " * indent_level
    return "\n".join(f"{indent_string}{line}" for line in text.splitlines())


def _format_special_object(obj: Any) -> str:
    """Format special objects like tool choice and response format."""
    if obj is None:
        return "None"
    elif obj is object():
        return "Not Set"
    elif isinstance(obj, bool):
        return "✅Enabled" if obj else "❌ Disabled"
    else:
        return str(obj)


def _format_stream_info(stream: bool, tool_choice: Any, response_format: Any) -> str:
    """Format stream, tool choice and response format information."""
    info = [
        "\n🦾 Configuration:",
        f"      Streaming  : {_format_special_object(stream)}",
        f"      Tool Mode  : {_format_special_object(tool_choice)}",
        f"      Response   : {_format_special_object(response_format)}",
    ]
    return "\n" + "\n".join(_indent(line, 1) for line in info)


def _format_output(output: Any) -> str:
    """Format different types of output into a string representation."""

    if output is None:
        return "None"
    elif output is object():
        return "Not Set"
    elif isinstance(output, str):
        lines = output.splitlines()
        if len(lines) > 1:
            width = max(len(line) for line in lines)
            border = "+" + "-" * (width + 2) + "+"
            formatted_lines = [border]
            for line in lines:
                formatted_lines.append(f"| {line:<{width}} |")
            formatted_lines.append(border)
            return "\n".join(formatted_lines)
        return output
    elif isinstance(output, BaseModel):
        json_str = output.model_dump_json(indent=2)
        return f"📋 Model Output:\n{_indent(json_str, 1)}"
    elif isinstance(output, (list, tuple)):
        if not output:
            return "[]"
        items = [f"• {item}" for item in output]
        return "\n" + "\n".join(_indent(item, 1) for item in items)
    elif isinstance(output, dict):
        if not output:
            return "{}"
        max_key_length = max(len(str(k)) for k in output.keys())
        items = [f"• {str(k):<{max_key_length}} : {_format_special_object(v)}" for k, v in output.items()]
        return "\n" + "\n".join(_indent(item, 1) for item in items)
    else:
        return _format_special_object(output)


def _format_stats(result: PrettyPrintable) -> str:
    """Format the statistics section of the result."""
    stats = [
        "\n📊 Statistics:",
        f"      Items     : {len(result.new_items)}",
        f"      Responses : {len(result.raw_responses)}",
        f"      Input GR  : {len(result.input_guardrail_results)}",
        f"      Output GR : {len(result.output_guardrail_results)}",
    ]
    return "\n" + "\n".join(_indent(stat, 1) for stat in stats)


def _format_agent_info(result: PrettyPrintable) -> str:
    """Format the agent information section."""
    if hasattr(result, 'is_complete'):
        info = [
            "\n👾 Agent Info:",
            f"      Name       : {result.current_agent.name}",
            f"      Turn       : {result.current_turn}/{result.max_turns}",
            f"      Status     : {'✅ Complete' if result.is_complete else '🔄 Running'}",
        ]
    else:
        info = [
            "\n👾 Agent Info:",
            f"      Last Agent : {result.last_agent.name}",
        ]
    return "\n" + "\n".join(_indent(line, 1) for line in info)


def _format_final_output(result: PrettyPrintable) -> str:
    """Format the final output section."""
    header = "\n\n  ✨ Final Output:\n"

    try:
        output = str(result.raw_responses[0].output[0])
        think_pattern = r'<think>(.*?)</think>(.*)'
        match = re.search(think_pattern, output, re.DOTALL)

        if match:
            reasoning = match.group(1).strip()
            final_result = match.group(2).strip()
            result_pattern = r"^([^']*?)(?:',\s*'type':.*)?$"
            final_result = re.match(result_pattern, final_result)
            if final_result:
                final_result = final_result.group(1).strip()
                reasoning = reasoning.encode().decode('unicode-escape')
                final_result = final_result.encode().decode('unicode-escape')

                sections = [
                    header,
                    "       🤔 REASONING:",
                    reasoning,
                    "       🎯 RESULT:",
                    final_result
                ]
                return "\n".join(sections) + "\n"

    except Exception as e:
        print(f"Error formatting final output: {e}")
        return ""


########################################################
#               Public Functions                       #
########################################################

def pretty_print_result(result: PrettyPrintable) -> str:
    """Pretty print a RunResult object."""
    output = f"🎯 {result.__class__.__name__}:"
    output += _format_agent_info(result)
    output += _format_stats(result)
    output += _format_stream_info(
        stream=hasattr(result, 'is_complete'),
        tool_choice=getattr(result, 'tool_choice', None),
        response_format=getattr(result, 'response_format', None)
    )
    output += _format_final_output(result)
    return output


def pretty_print_run_result_streaming(result: PrettyPrintable) -> str:
    """Pretty print a RunResultStreaming object."""
    return pretty_print_result(result)


def format_json_response(response: dict[str, Any]) -> str:
    """Format a JSON response to be more readable."""

    role = response.get("role", "")
    content = response.get("content", "")

    role_emoji = {
        "assistant": "🤖",
        "user": "👤",
        "system": "⚙️",
        "function": "🔧"
    }.get(role, "📝")

    header = f"{role_emoji} {role.title()}"
    header_border = "═" * (len(header) + 4)
    header_text = f"╔{header_border}╗\n║ {header} ║\n╚{header_border}╝"

    formatted_lines = []
    sections = content.split("</think>")
    if len(sections) > 1:
        thinking = sections[0].replace("<think>", "").strip()
        if thinking:
            formatted_lines.append("💭 Thinking:")
            for line in thinking.split("\n"):
                if line.strip():
                    formatted_lines.append(f"  {line.strip()}")
            formatted_lines.append("")

    final_content = sections[-1].strip()
    final_content = final_content.replace("**", "").replace("*", "")
    words = final_content.split()
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > 78:
            if current_line:
                formatted_lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1

    if current_line:
        formatted_lines.append(" ".join(current_line))

    content_text = "\n".join(formatted_lines)
    bottom_border = "═" * 80
    return f"{header_text}\n\n{_indent(content_text, 1)}\n\n╔{bottom_border}╗"
