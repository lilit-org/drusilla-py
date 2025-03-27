#!/usr/bin/env python3

"""
This script demonstrates the agents-as-tools pattern where a frontline agent selects
translation agents to handle user messages.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.agents import Agent, Runner
from src.util._pretty_print import pretty_print_result
from src.util._client import setup_client
from src.util._exceptions import GenericError

from src.util._constants import SUPPORTED_LANGUAGES


########################################################
#           Agent Creation                             #
########################################################

def create_agents() -> Agent:

    orchestrator_agent = Agent(
        name="Agent World Traveler",
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


########################################################
#           Agent Runner                               #
########################################################

def run_agent() -> str | None:

    try:
        setup_client()
        agent = create_agents()

        msg = input("\n✅ Enter text to translate and target languages: ")
        result = Runner.run_sync(agent, msg)
        print(pretty_print_result(result))
    except Exception as e:
        raise GenericError(e)


if __name__ == "__main__":
    run_agent()