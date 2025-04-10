#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.agent_v1 import AgentV1 as Agent
from src.network.client import setup_client
from src.runners.run import Runner
from src.util.constants import config
from src.util.exceptions import AgentExecutionError
from src.util.print import pretty_print_result, pretty_print_result_stats

########################################################
#           Agent Creation                             #
########################################################


def create_agents() -> Agent:
    return Agent(
        name="Agent World Traveler",
        instructions=(
            "You are a cool special robot who coordinates translation requests."
            "Use appropriate translation swords based on requested languages."
        ),
        swords=[
            Agent(
                name=f"{lang_code.upper()} Translator",
                instructions=f"Translate English text to {lang_code.upper()}",
                orbs_description=f"English to {lang_code.upper()} translator",
            ).as_sword(
                sword_name=f"translate_to_{lang_code.lower()}",
                sword_description=f"Translate text to {lang_code.upper()}",
            )
            for lang_code in config.SUPPORTED_LANGUAGES
        ],
    )


########################################################
#           Agent Runner                               #
########################################################


def run_agent() -> str | None:
    try:
        setup_client()
        agent = create_agents()
        msg = input("\n❓ Enter text to translate: ")
        result = Runner.run_sync(agent, msg)
        print(pretty_print_result(result))
        print(pretty_print_result_stats(result))
    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    run_agent()
