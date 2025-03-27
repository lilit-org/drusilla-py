#!/usr/bin/env python3

"""
A weather information agent that demonstrates the usage of tools this framework.
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))

from src.agents import Agent, Runner
from src.util._client import setup_client
from src.util._env import get_env_var
from src.util._exceptions import GenericError
from src.util._pretty_print import pretty_print_result
from src.util._tool import function_tool

########################################################
#           Constants                                  #
########################################################

CACHE_SIZE = int(get_env_var("CACHE_SIZE", "128"))
SUMMER_TEMP_THRESHOLD = 25

########################################################
#           Models and Tools                           #
########################################################


class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str
    is_summer: bool


@function_tool
@lru_cache(maxsize=CACHE_SIZE)
def get_weather(city: str) -> Weather:
    print(f"Getting weather for {city}")
    return Weather(
        city=city, temperature_range="0-30C", conditions="Clear skies", is_summer=False
    )


def _parse_temperature_range(temp_range: str) -> tuple[int, int]:
    try:
        min_temp, max_temp = temp_range.split("-")
        return int(min_temp), int(max_temp.rstrip("C"))
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid temperature range format: {temp_range}") from e


@function_tool
@lru_cache(maxsize=CACHE_SIZE)
def is_summer(city: str) -> bool:
    weather = get_weather(city)
    _, max_temp = _parse_temperature_range(weather.temperature_range)
    return max_temp >= SUMMER_TEMP_THRESHOLD


########################################################
#           Agent Creation                             #
########################################################


@lru_cache(maxsize=1)
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


def run_agent() -> Optional[str]:
    try:
        setup_client()
        agent = create_agent()
        msg = input("\n❓ Enter a city to check the weather: ").strip()
        result = Runner.run_sync(agent, msg)
        print(pretty_print_result(result))
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise GenericError(e) from e


if __name__ == "__main__":
    run_agent()
