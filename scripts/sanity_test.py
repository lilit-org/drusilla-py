#!/usr/bin/env python3

"""
This script demonstrates basic usage of the DeepSeekClient and Agent classes.
"""

import sys
import httpx
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.util._client import DeepSeekClient
from src.agents.run import Runner
from src.agents.agent import Agent
from src import set_default_model_client, set_default_model_api
from src.util._pretty_print import pretty_print_result


def setup_client() -> DeepSeekClient:
    """Set up and configure the DeepSeek client with optimal settings."""
    client = DeepSeekClient(
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=30.0, read=90.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    )
    set_default_model_client(client)
    set_default_model_api("chat_completions")
    return client


def create_agent() -> Agent:
    """Create a configured agent instance."""
    return Agent(
        name="Agent Mulder",
        instructions="You are a cool special agent robot"
    )


def main() -> str | None:
    """Run the sanity test and return the result."""
    try:
        setup_client()
        agent = create_agent()
        
        result = Runner.run_sync(
            agent,
            "Write a haiku about love in the cypherpunk world."
        )
        return pretty_print_result(result)
    except httpx.HTTPError as e:
        print(f"HTTP error occurred: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error running sanity test: {e}", file=sys.stderr)
    return None


if __name__ == "__main__":
    if output := main():
        print(output)
