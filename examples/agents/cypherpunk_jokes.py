#!/usr/bin/env python3

"""
This script demonstrates the streaming functionality of the agent framework,
showing how to handle different types of stream events in real-time.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.agent import Agent
from src.agents.run import Runner
from src.util._client import setup_client
from src.util._exceptions import AgentExecutionError
from src.util._print import pretty_print_result_stats, pretty_print_result_stream

########################################################
#           Agent Creation                             #
########################################################


def create_agent() -> Agent:
    return Agent(
        name="Agent Cypherpunk Joker",
        instructions=(
            "You are a cool special robot who tells jokes. Follow these steps exactly:\n"
            "1. First, parse the number of jokes requested from the input message\n"
            "2. After getting the number N, tell exactly N jokes\n"
            "3. Each joke should be on its own line\n"
            "4. Make the jokes cyberpunk-themed and entertaining\n"
            "5. Number each joke clearly"
        ),
    )


########################################################
#           Private methods
########################################################


def _get_number_of_jokes() -> int:
    while True:
        try:
            n = int(input("\n❓ How many jokes would you like to hear? (1-10): "))
            if 1 <= n <= 10:
                return n
            print("❌ Please enter a number between 1 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")


async def _handle_stream_events(result, num_jokes):
    try:
        async for event in result.stream_events():

            if event.type == "agent_updated_stream_event":
                print(f"\n✅ {event.new_agent.name} is telling {num_jokes} jokes...")

            if event.type == "raw_response_event":
                if hasattr(event.data, "type") and event.data.type == "content_part.done":
                    if isinstance(event.data.part, dict) and event.data.part.get("text"):
                        print(
                            pretty_print_result_stream(event.data.part["text"]),
                            end="",
                            flush=True,
                        )

            await asyncio.sleep(0)
        print(pretty_print_result_stats(result))
    except asyncio.CancelledError as e:
        raise AgentExecutionError("Stream processing was cancelled") from e


########################################################
#           Agent Runner                               #
########################################################


async def run_agent():
    try:
        setup_client()
        agent = create_agent()
        num_jokes = _get_number_of_jokes()

        result = await Runner.run_streamed(
            agent,
            f"Tell me exactly {num_jokes} cypherpunk jokes!",
        )
        await _handle_stream_events(result, num_jokes)
    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    asyncio.run(run_agent())
