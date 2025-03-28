#!/usr/bin/env python3

"""
A weather information agent that demonstrates the usage of tools this framework.
"""

import sys
from functools import lru_cache
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.agent import Agent
from src.agents.run import Runner
from src.util._client import setup_client
from src.util._env import get_env_var
from src.util._exceptions import AgentExecutionError, UsageError
from src.util._pretty_print import pretty_print_result
from src.util._tool import function_tool

########################################################
#           Tools                                      #
########################################################


@function_tool
@lru_cache(maxsize=int(get_env_var("CACHE_SIZE", "128")))
def get_weather(city: str) -> dict:
    print(f"Getting weather for {city}")
    return {
        "city": city,
        "temperature_range": "0-30C",
        "conditions": "Clear skies",
        "is_summer": False,
    }


@function_tool
@lru_cache(maxsize=int(get_env_var("CACHE_SIZE", "128")))
def is_summer(city: str) -> bool:
    weather = get_weather(city)
    try:
        _, max_temp = map(int, weather["temperature_range"].split("-")[1].rstrip("C"))
        return max_temp >= 25
    except (ValueError, AttributeError) as e:
        raise UsageError(
            f"Invalid temperature range format: {weather['temperature_range']}"
        ) from e


########################################################
#           Agent Creation                             #
########################################################


@lru_cache(maxsize=int(get_env_var("CACHE_SIZE", "128")))
def create_agent() -> Agent:
    return Agent(
        name="Agent Summer Chaser",
        instructions=(
            "You are a cool special robot who provides accurate weather information "
            "and tells whether it's summer or not. For EVERY request: "
            "1. Use the weather tool to fetch weather data for the requested city "
            "2. ALWAYS use the is_summer tool to check if it feels like summer "
            "3. Present both the weather information AND whether it's summer in your response."
        ),
        tools=[get_weather, is_summer],
    )


########################################################
#           Agent Runner                               #
########################################################


def run_agent() -> str | None:
    try:
        setup_client()
        agent = create_agent()
        msg = input("\n❓ Enter a city to check the weather: ").strip()
        result = Runner.run_sync(agent, msg)
        print(pretty_print_result(result))
    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    run_agent()
