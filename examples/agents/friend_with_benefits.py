#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.agent_v1 import AgentV1 as Agent
from src.gear.orbs import OrbsInputData, orbs
from src.gear.sword import function_sword
from src.network.client import setup_client
from src.runners.run import Runner
from src.util.exceptions import AgentExecutionError
from src.util.print import pretty_print_result, pretty_print_result_stats
from src.util.types import RunContextWrapper

########################################################
#              Swords and Filters
########################################################


@function_sword
def random_number() -> int:
    return random.randint(3, 15)


def orbs_message_filter(orbs_message_data: OrbsInputData) -> OrbsInputData:
    """Filter the message history to remove swords and keep only relevant history."""
    # Filter out sword-related items from the history
    if isinstance(orbs_message_data.input_history, tuple):
        filtered_history = tuple(
            item
            for item in orbs_message_data.input_history[2:]
            if not (isinstance(item, dict) and item.get("type") == "sword")
        )
    else:
        filtered_history = orbs_message_data.input_history

    return OrbsInputData(
        input_history=filtered_history,
        pre_orbs_items=tuple(orbs_message_data.pre_orbs_items),
        new_items=tuple(orbs_message_data.new_items),
    )


########################################################
#               Agents Creation
########################################################


async def transfer_to_third_agent(ctx: RunContextWrapper[Any]) -> None:
    """Transfer to the third agent."""


async def transfer_to_second_agent(ctx: RunContextWrapper[Any]) -> None:
    """Transfer to the second agent."""


third_agent = Agent(
    name="Assistant Three",
    instructions="Replace one word in the sentence received from agent two with 'love' in a way that makes sense or is entertaining.",
    orbs_description="Replace one word in the input sentence with the word 'love'.",
)

second_agent = Agent(
    name="Agent Two",
    instructions="Create a sentence about the cypherpunk world with number of words exactly equal to the input number from agent one.",
    orbs=[orbs(third_agent, input_filter=orbs_message_filter)(transfer_to_third_agent)],
    orbs_description="Create sentences about the cypherpunk world.",
)

first_agent = Agent(
    name="Agent One",
    instructions="Generate a random between 3 and 15.",
    swords=[random_number],
)

########################################################
#               Agent Runner
########################################################


async def run_agent_chain(agent: Agent, input_messages: Sequence[dict]) -> None:
    result = await Runner.run(agent, input=input_messages)
    print(pretty_print_result_stats(result))
    print(pretty_print_result(result, show_reasoning=False))
    return result


async def run_agent():
    try:
        setup_client()

        result = await run_agent_chain(
            first_agent, [{"content": "Generate a random number between 3 and 15.", "role": "user"}]
        )

        result = await run_agent_chain(
            second_agent,
            [item.input_item for item in result.new_items]
            + [
                {
                    "content": "Create a sentence about the cypherpunk world with the specified word count.",
                    "role": "user",
                }
            ],
        )

        await run_agent_chain(
            third_agent,
            [item.input_item for item in result.new_items]
            + [
                {
                    "content": "Replace one word with 'love' in a way that makes sense or is entertaining.",
                    "role": "user",
                }
            ],
        )

    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    asyncio.run(run_agent())
