#!/usr/bin/env python3

"""
This script demonstrates the agents-as-swords pattern where a frontline agent selects
translation agents to handle user messages.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from functools import lru_cache

from src.agents.agent import Agent
from src.agents.run import Runner
from src.gear.sword import function_sword
from src.util._client import setup_client
from src.util._constants import LRU_CACHE_SIZE
from src.util._exceptions import AgentExecutionError, UsageError
from src.util._print import pretty_print_result, pretty_print_result_stats

########################################################
#           Swords
########################################################


@function_sword
@lru_cache(maxsize=LRU_CACHE_SIZE)
def get_weather(city: str) -> dict:
    print(f"Getting weather for {city}")
    return {
        "city": city,
        "temperature_range": "0-30C",
        "conditions": "Clear skies",
        "is_summer": False,
    }


@function_sword
@lru_cache(maxsize=LRU_CACHE_SIZE)
def is_summer(city: str) -> bool:
    weather = get_weather(city)
    try:
        _, max_temp = map(int, weather["temperature_range"].split("-")[1].rstrip("C"))
        return max_temp >= 25
    except (ValueError, AttributeError) as e:
        raise UsageError(f"Invalid temperature range format: {weather['temperature_range']}") from e


########################################################
#           Agent Creation
########################################################


@lru_cache(maxsize=LRU_CACHE_SIZE)
def create_agent() -> Agent:
    return Agent(
        name="Agent Summer Chaser",
        instructions=(
            "You are a cool special robot who provides accurate weather information "
            "and tells whether it's summer or not. For EVERY request: "
            "1. Use the weather sword to fetch weather data for the requested city "
            "2. ALWAYS use the is_summer sword to check if it feels like summer "
            "3. Present both the weather information AND whether it's summer in your response."
        ),
        swords=[get_weather, is_summer],
    )


########################################################
#           Agent Runne
########################################################


def run_agent() -> str | None:
    try:
        setup_client()
        agent = create_agent()
        msg = input("\n❓ Enter a city to check the weather: ").strip()
        result = Runner.run_sync(agent, msg)
        print(pretty_print_result(result))
        print(pretty_print_result_stats(result))
    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    run_agent()
