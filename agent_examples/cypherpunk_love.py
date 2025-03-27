#!/usr/bin/env python3

"""
This script demonstrates basic usage of the DeepSeekClient and Agent classes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.agents import Agent, Runner
from src.util._pretty_print import pretty_print_result
from src.util._client import setup_client
from src.util._exceptions import GenericError


########################################################
#           Agent Creation                             #
########################################################

def create_agent() -> Agent:

    return Agent(
        name="Agent Dr. Love",
        instructions="You are a cool special robot who loves"
    )


########################################################
#           Agent Runner                               #
########################################################

def run_agent() -> str | None:

    try:
        setup_client()
        agent = create_agent()
        
        result = Runner.run_sync(
            agent,
            "Write a haiku about love in the cypherpunk world."
        )
        print(pretty_print_result(result))
    except Exception as e:
        raise GenericError(e)


if __name__ == "__main__":
    run_agent()
