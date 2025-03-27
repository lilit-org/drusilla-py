import re
from re import Pattern
from typing import Any
from ._exceptions import GenericError
from ._items import ModelResponse
from ._result import RunResult

########################################################
#               Compiled Regex Patterns                #
########################################################

THINK_PATTERN: Pattern[str] = re.compile(r'<think>(.*?)</think>(.*)', re.DOTALL)
RESULT_PATTERN: Pattern[str] = re.compile(r"^([^']*?)(?:',\s*'type':\s*'output_text',\s*'annotations':\s*\[\])?$")
TEXT_PATTERN: Pattern[str] = re.compile(r"'text':\s*'([^']*)'")

########################################################
#               Final Output Section                  #
########################################################

def _decode_unicode_escape(text: str) -> str:
    return text.encode().decode('unicode-escape')


def _format_final_output(raw_response: ModelResponse) -> str:

    try:
        output = str(raw_response.output[0])
        if match := THINK_PATTERN.search(output):
            reasoning = _decode_unicode_escape(match.group(1).strip())
            final_result = match.group(2).strip()
            if result_match := RESULT_PATTERN.match(final_result):
                final_result = _decode_unicode_escape(result_match.group(1).strip())
            else:
                final_result = _decode_unicode_escape(final_result.strip())
            return f"\n\n✅ REASONING:\n{reasoning}\n\n✅ RESULT:\n{final_result}\n"

        if result_match := RESULT_PATTERN.match(output):
            final_result = _decode_unicode_escape(result_match.group(1).strip())
            return f"\n\n✅ RESULT:\n{final_result}\n"

        if text_match := TEXT_PATTERN.search(output):
            final_result = _decode_unicode_escape(text_match.group(1).strip())
            return f"\n\n✅ RESULT:\n{final_result}\n"

        output = re.sub(r"',\s*'type':\s*'output_text',\s*'annotations':\s*\[\]", "", output)
        final_result = _decode_unicode_escape(output.strip("'").strip())
        return f"\n\n✅ RESULT:\n{final_result}\n"

    except GenericError as e:
        print(f"Error formatting final output: {e}")
        return ""


########################################################
#               Agent Info Section                    #
########################################################

def _indent(text: str, indent_level: int) -> str:
    indent_string = "  " * indent_level
    return "\n".join(f"{indent_string}{line}" for line in text.splitlines())


def _format_agent_info(result: RunResult) -> str:
    if hasattr(result, 'is_complete'):
        info = [
            "\n👾 Agent Info:",
            f"      Name   → {result.current_agent.name}",
            f"      Turn   → {result.current_turn}/{result.max_turns}",
            f"      Status → {'✔️ Complete' if result.is_complete else '🟡 Running'}",
        ]
    else:
        info = [
            "\n👾 Agent Info:",
            f"      Last Agent → {result.last_agent.name}",
        ]
    return "\n" + "\n".join(_indent(line, 1) for line in info)


########################################################
#               Stats Section                         #
########################################################

def _format_stats(result: RunResult) -> str:
    stats = [
        "\n📊 Statistics:",
        f"      Items     → {len(result.new_items)}",
        f"      Responses → {len(result.raw_responses)}",
        f"      Input GR  → {len(result.input_guardrail_results)}",
        f"      Output GR → {len(result.output_guardrail_results)}",
    ]
    return "\n" + "\n".join(_indent(stat, 1) for stat in stats)


########################################################
#               Stream Info Section                   #
########################################################

def _format_stream_object(obj: Any) -> str:
    if obj is None or obj is object():
        return "None"
    elif isinstance(obj, bool):
        return "✔️ Enabled" if obj else "❌ Disabled"
    else:
        return str(obj)


def _format_stream_info(stream: bool, tool_choice: Any) -> str:
    info = [
        "\n🦾 Configuration:",
        f"      Streaming → {_format_stream_object(stream)}",
        f"      Tool Mode → {_format_stream_object(tool_choice)}",
    ]
    return "\n" + "\n".join(_indent(line, 1) for line in info)


########################################################
#               Public Main Function                  #
########################################################

def pretty_print_result(result: RunResult) -> str:

    parts = [
        f"✅ {result.__class__.__name__}:",
        _format_agent_info(result),
        _format_stats(result),
        _format_stream_info(
            stream=hasattr(result, 'is_complete'),
            tool_choice=getattr(result, 'tool_choice', None)
        ),
        _format_final_output(result.raw_responses[0])
    ]

    return "".join(parts)
