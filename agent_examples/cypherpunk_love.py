#!/usr/bin/env python3

"""
This script demonstrates basic usage of the DeepSeekClient and Agent classes.
"""

import sys
import httpx
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.agents import Agent, Runner
from src.util._pretty_print import pretty_print_result
from src.util._client import setup_client


def create_agent() -> Agent:

    return Agent(
        name="Agent Dr. Love",
        instructions="You are a cool special robot who loves"
    )


def main() -> str | None:

    try:
        setup_client()
        agent = create_agent()
        
        result = Runner.run_sync(
            agent,
            "Write a haiku about love in the cypherpunk world."
        )
        print(pretty_print_result(result))
    except httpx.HTTPError as e:
        print(f"HTTP error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running poetry generator: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
