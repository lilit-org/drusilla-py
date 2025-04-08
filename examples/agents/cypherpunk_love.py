#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.agent_v1 import AgentV1 as Agent
from src.network.client import setup_client
from src.runners.run import Runner
from src.util.exceptions import AgentExecutionError
from src.util.print import pretty_print_result, pretty_print_result_stats

########################################################
#           Agent Creation                             #
########################################################


def create_agent() -> Agent:
    return Agent(name="Agent Dr. Love", instructions="You are a cool special robot who loves")


########################################################
#           Agent Runner                               #
########################################################


def run_agent() -> str | None:
    try:
        setup_client()
        agent = create_agent()
        result = Runner.run_sync(agent, "Write a haiku about love in the cypherpunk world.")
        print(pretty_print_result(result))
        print(pretty_print_result_stats(result))
    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    run_agent()
