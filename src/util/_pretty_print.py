import re
from re import Pattern
from typing import Any, Protocol

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
#               Compiled Regex Patterns                #
########################################################

THINK_PATTERN: Pattern[str] = re.compile(r'<think>(.*?)</think>(.*)', re.DOTALL)
RESULT_PATTERN: Pattern[str] = re.compile(r"^([^']*?)(?:',\s*'type':\s*'output_text',\s*'annotations':\s*\[\])?$")
TEXT_PATTERN: Pattern[str] = re.compile(r"'text':\s*'([^']*)'")
METADATA_PATTERN: Pattern[str] = re.compile(r"',\s*'type':\s*'output_text',\s*'annotations':\s*\[\]$")

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
        f"      Streaming → {_format_special_object(stream)}",
        f"      Tool Mode → {_format_special_object(tool_choice)}",
    ]
    return "\n" + "\n".join(_indent(line, 1) for line in info)


def _format_stats(result: PrettyPrintable) -> str:
    """Format the statistics section of the result."""
    stats = [
        "\n📊 Statistics:",
        f"      Items     → {len(result.new_items)}",
        f"      Responses → {len(result.raw_responses)}",
        f"      Input GR  → {len(result.input_guardrail_results)}",
        f"      Output GR → {len(result.output_guardrail_results)}",
    ]
    return "\n" + "\n".join(_indent(stat, 1) for stat in stats)


def _format_agent_info(result: PrettyPrintable) -> str:
    """Format the agent information section."""
    if hasattr(result, 'is_complete'):
        info = [
            "\n👾 Agent Info:",
            f"      Name   → {result.current_agent.name}",
            f"      Turn   → {result.current_turn}/{result.max_turns}",
            f"      Status → {'✅ Complete' if result.is_complete else '🔄 Running'}",
        ]
    else:
        info = [
            "\n👾 Agent Info:",
            f"      Last Agent → {result.last_agent.name}",
        ]
    return "\n" + "\n".join(_indent(line, 1) for line in info)


def _decode_unicode_escape(text: str) -> str:
    """Decode unicode escape sequences in text."""
    return text.encode().decode('unicode-escape')


def _format_final_output(result: PrettyPrintable) -> str:
    """Format the final output section."""
    try:
        output = str(result.raw_responses[0].output[0])
        match = THINK_PATTERN.search(output)

        if match:
            reasoning = match.group(1).strip()
            final_result = match.group(2).strip()
            result_match = RESULT_PATTERN.match(final_result)

            if result_match:
                final_result = result_match.group(1).strip()
            else:
                final_result = final_result.strip()

            reasoning = _decode_unicode_escape(reasoning)
            final_result = _decode_unicode_escape(final_result)

            return f"\n\n✅ REASONING:\n{reasoning}\n\n✅ RESULT:\n{final_result}\n"

        result_match = RESULT_PATTERN.match(output)
        if result_match:
            final_result = result_match.group(1).strip()
            final_result = _decode_unicode_escape(final_result)
            return f"\n\n✅ RESULT:\n{final_result}\n"

        text_match = TEXT_PATTERN.search(output)
        if text_match:
            final_result = text_match.group(1).strip()
            final_result = _decode_unicode_escape(final_result)
            return f"\n\n✅ RESULT:\n{final_result}\n"

        try:
            text_parts = re.findall(r"'text':\s*'([^']*)'", output)
            if text_parts:
                final_result = text_parts[0]
                final_result = _decode_unicode_escape(final_result)
                return f"\n\n✅ RESULT:\n{final_result}\n"
        except:
            pass

        if "', 'type': 'output_text', 'annotations': []" in output:
            parts = output.split("', 'type': 'output_text', 'annotations': []")
            if parts:
                output = parts[0]

        output = re.sub(r"',\s*'type':\s*'output_text',\s*'annotations':\s*\[\]", "", output)
        output = output.strip("'").strip()

        final_result = _decode_unicode_escape(output)
        return f"\n\n✅ RESULT:\n{final_result}\n"

    except Exception as e:
        print(f"Error formatting final output: {e}")
        return ""


def _wrap_text(text: str, max_width: int = 78) -> list[str]:
    """Wrap text to specified width."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_width:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1

    if current_line:
        lines.append(" ".join(current_line))

    return lines


########################################################
#               Public Functions                       #
########################################################

def pretty_print_result(result: PrettyPrintable) -> str:
    """Pretty print a RunResult object."""
    parts = [
        f"✅ {result.__class__.__name__}:",
        _format_agent_info(result),
        _format_stats(result),
        _format_stream_info(
            stream=hasattr(result, 'is_complete'),
            tool_choice=getattr(result, 'tool_choice', None),
            response_format=getattr(result, 'response_format', None)
        ),
        _format_final_output(result)
    ]
    return "".join(parts)


def pretty_print_run_result_streaming(result: PrettyPrintable) -> str:
    """Pretty print a RunResultStreaming object."""
    return pretty_print_result(result)


def format_json_response(response: dict[str, Any]) -> str:
    """Format a JSON response to be more readable."""
    role = response.get("role", "")
    content = response.get("content", "")
    header = f"{role.title()}"
    header_border = "═" * (len(header) + 4)
    header_text = f"╔{header_border}╗\n║ {header} ║\n╚{header_border}╝"

    formatted_lines = []
    sections = content.split("</think>")

    if len(sections) > 1:
        thinking = sections[0].replace("<think>", "").strip()
        if thinking:
            formatted_lines.append("💭 Thinking:")
            formatted_lines.extend(f"  {line.strip()}" for line in thinking.split("\n") if line.strip())
            formatted_lines.append("")

    final_content = sections[-1].strip()
    final_content = final_content.replace("**", "").replace("*", "")
    formatted_lines.extend(_wrap_text(final_content))

    content_text = "\n".join(formatted_lines)
    bottom_border = "═" * 80
    return f"{header_text}\n\n{_indent(content_text, 1)}\n\n╔{bottom_border}╗"
