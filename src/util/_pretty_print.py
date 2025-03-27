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


def _format_stream_info(stream: bool, tool_choice: Any) -> str:
    """Format stream and tool choice information."""
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
            reasoning = _decode_unicode_escape(match.group(1).strip())
            final_result = match.group(2).strip()
            result_match = RESULT_PATTERN.match(final_result)
            final_result = _decode_unicode_escape(result_match.group(1).strip() if result_match else final_result.strip())
            return f"\n\n✅ REASONING:\n{reasoning}\n\n✅ RESULT:\n{final_result}\n"

        result_match = RESULT_PATTERN.match(output)
        if result_match:
            final_result = _decode_unicode_escape(result_match.group(1).strip())
            return f"\n\n✅ RESULT:\n{final_result}\n"

        text_match = TEXT_PATTERN.search(output)
        if text_match:
            final_result = _decode_unicode_escape(text_match.group(1).strip())
            return f"\n\n✅ RESULT:\n{final_result}\n"

        output = re.sub(r"',\s*'type':\s*'output_text',\s*'annotations':\s*\[\]", "", output)
        final_result = _decode_unicode_escape(output.strip("'").strip())
        return f"\n\n✅ RESULT:\n{final_result}\n"

    except Exception as e:
        print(f"Error formatting final output: {e}")
        return ""


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
            tool_choice=getattr(result, 'tool_choice', None)
        ),
        _format_final_output(result)
    ]
    return "".join(parts)
