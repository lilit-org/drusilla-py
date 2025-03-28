#!/usr/bin/env python3

"""
A chameleon-like agent that changes its personality and response style based on context.
"""

import sys
from enum import Enum
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.agents.agent import Agent
from src.agents.run import Runner
from src.util._client import setup_client
from src.util._constants import DEFAULT_MAX_TURNS
from src.util._env import get_env_var
from src.util._exceptions import AgentExecutionError
from src.util._pretty_print import pretty_print_result
from src.util._run_context import RunContextWrapper

########################################################
#           Constants
########################################################

MAX_TURNS = int(get_env_var("MAX_TURNS", str(DEFAULT_MAX_TURNS)))

########################################################
#           Style instructions
########################################################


class Style(Enum):
    HAIKU = "haiku"
    PIRATE = "pirate"
    ROBOT = "robot"
    POET = "poet"
    SHAKESPEARE = "shakespeare"
    VALLEY_GIRL = "valley_girl"
    DETECTIVE = "detective"
    PROFESSOR = "professor"
    CHEF = "chef"
    FORTUNE_TELLER = "fortune_teller"
    STONER = "stoner"


STYLE_INSTRUCTIONS = {
    Style.HAIKU: (
        "Only respond in haikus. Each response must follow the 5-7-5 syllable pattern. "
        f"Keep responses concise and nature-themed when possible. Limit to {MAX_TURNS} turns."
    ),
    Style.PIRATE: (
        "Respond as a pirate. Use phrases like 'arr matey', 'aye', 'shiver me timbers', "
        "and 'yo ho ho'. Speak in a rough, adventurous tone."
    ),
    Style.ROBOT: (
        "Respond as a robot. Say 'beep boop' frequently. Use mechanical language, "
        "binary references, and speak in a precise, calculated manner."
    ),
    Style.POET: (
        "Respond in rhyming couplets with a romantic and flowery style. Use metaphors, "
        "similes, and poetic devices. Each response should be lyrical and emotionally resonant."
    ),
    Style.SHAKESPEARE: (
        "Respond in Shakespearean style. Use archaic language like 'thee', 'thou', "
        "'dost', and 'hath'. Include Shakespearean metaphors and dramatic flair."
    ),
    Style.VALLEY_GIRL: (
        "Respond like a valley girl. Use 'like', 'totally', 'oh my gosh', and 'whatever' "
        "frequently. Speak with enthusiasm and include modern slang."
    ),
    Style.DETECTIVE: (
        "Respond like a hard-boiled noir detective. Use metaphors about rain, shadows, "
        "and cigarettes. Be cynical and speak in a world-weary tone."
    ),
    Style.PROFESSOR: (
        "Respond like an academic professor. Use scholarly language, cite sources, "
        "and explain concepts thoroughly. Maintain a formal, educational tone."
    ),
    Style.CHEF: (
        "Respond like a passionate chef. Use cooking metaphors, food-related expressions, "
        "and culinary terminology. Speak with enthusiasm about flavors and techniques."
    ),
    Style.FORTUNE_TELLER: (
        "Respond like a mystical fortune teller. Use mystical language, crystal ball "
        "references, and make cryptic predictions. Speak in a mysterious, ethereal tone."
    ),
    Style.STONER: (
        "Respond like a laid-back stoner. Use phrases like 'dude', 'man', 'whoa', "
        "and 'far out'. Speak in a relaxed, mellow tone with lots of 'like' and 'you know'. "
        "Make philosophical observations about life and reality."
    ),
}


def get_style_instructions(
    run_context: RunContextWrapper[Style], _: Agent[Style]
) -> str:
    return STYLE_INSTRUCTIONS[run_context.context]


def display_style_options():
    print("\n✅ Available styles:")
    for i, style in enumerate(Style, 1):
        print(f"    {i}. {style.value.replace('_', ' ').title()}")
    print()


def get_style_choice() -> Style:
    while True:
        try:
            choice = int(input("❓ Enter the number of your desired style: "))
            if 1 <= choice <= len(Style):
                return list(Style)[choice - 1]
            print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number...")


########################################################
#           Agent Creation                             #
########################################################


def create_agent() -> Agent[Style]:
    return Agent(
        name="Agent Dissociative Identity",
        instructions=get_style_instructions,
    )


########################################################
#           Agent Runner                               #
########################################################


def run_agent():
    try:
        setup_client()
        agent = create_agent()

        display_style_options()
        style = get_style_choice()
        print(f"\n✅ Using style: {style.value.replace('_', ' ').title()}")
        print(f"✅ Style description: {STYLE_INSTRUCTIONS[style]}\n")

        msg = input("❓ Enter your message: ").strip()
        result = Runner.run_sync(agent, msg, context=style, max_turns=MAX_TURNS)
        print(pretty_print_result(result))
    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    run_agent()
