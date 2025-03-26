#!/usr/bin/env python3

"""
Sanity test script for the DeepSeek agent framework.
This script demonstrates basic usage of the DeepSeekClient and Agent classes
by running a simple task of generating a haiku.
"""

import os
import sys
import httpx
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.util._client import DeepSeekClient
from src.agents.run import Runner
from src.agents.agent import Agent
from src import set_default_model_client, set_default_model_api


def setup_client() -> DeepSeekClient:
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
    return Agent(
        name="Agent Mulder",
        instructions="You are a cool special agent robot"
    )


def main() -> Optional[str]:
    try:
        setup_client()
        agent = create_agent()
        
        result = Runner.run_sync(
            agent,
            "Write a haiku about love in the cypherpunk world."
        )
        return result.final_output
    except Exception as e:
        print(f"Error running sanity test: {e}", file=sys.stderr)


if __name__ == "__main__":
    output = main()
    if output:
        print(output)
