#!/usr/bin/env python3

"""
This script demonstrates the agents-as-tools pattern where a frontline agent selects
translation agents to handle user messages.
"""

import sys
import httpx
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.agents import Agent, Runner
from src.util._pretty_print import pretty_print_result
from src.util._client import setup_client

from src.util._constants import SUPPORTED_LANGUAGES


def create_agents() -> Agent:

    orchestrator_agent = Agent(
        name="World Traveler",
        instructions=(
            "Coordinate translation requests using provided tools. "
            "Use appropriate translation tools based on requested languages."
        ),
        tools=[
            Agent(
                name=f"{lang_name} Translator",
                instructions=f"Translate English text to {lang_name}",
                handoff_description=f"English to {lang_name} translator",
            ).as_tool(
                tool_name=f"translate_to_{lang_key.lower()}",
                tool_description=f"Translate text to {lang_name}",
            )
            for lang_key, lang_name in SUPPORTED_LANGUAGES.items()
        ],
    )

    return orchestrator_agent


def main() -> str | None:

    try:
        setup_client()
        orchestrator_agent = create_agents()
        msg = input("\n✅ Enter text to translate and target languages: ")
        result = Runner.run_sync(orchestrator_agent, msg)
        return pretty_print_result(result)
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Translation service error: {e}", file=sys.stderr)


if __name__ == "__main__":
    if output := main():
        print(output)
