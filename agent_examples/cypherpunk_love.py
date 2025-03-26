#!/usr/bin/env python3

"""
This script demonstrates basic usage of the DeepSeekClient and Agent classes.
"""

import sys
import httpx
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))

from src.util._client import DeepSeekClient
from src.agents.run import Runner
from src.agents.agent import Agent
from src import set_default_model_client, set_default_model_api
from src.util._pretty_print import pretty_print_result
from src.util._constants import (
    HTTP_TIMEOUT_TOTAL,
    HTTP_TIMEOUT_CONNECT,
    HTTP_TIMEOUT_READ,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_MAX_CONNECTIONS
)


def setup_client() -> DeepSeekClient:
    """Set up and configure the DeepSeek client with optimal settings."""
    client = DeepSeekClient(
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(
                HTTP_TIMEOUT_TOTAL,
                connect=HTTP_TIMEOUT_CONNECT,
                read=HTTP_TIMEOUT_READ
            ),
            limits=httpx.Limits(
                max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
                max_connections=HTTP_MAX_CONNECTIONS
            )
        )
    )
    set_default_model_client(client)
    set_default_model_api("chat_completions")
    return client


def create_agent() -> Agent:
    """Create a configured agent instance."""
    return Agent(
        name="Agent Dr. Love",
        instructions="You are a cool special robot who loves"
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
        print(pretty_print_result(result))
    except httpx.HTTPError as e:
        print(f"HTTP error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running poetry generator: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
