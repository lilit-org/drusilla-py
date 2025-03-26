#!/usr/bin/env python3

"""
This script demonstrates the agents-as-tools pattern where a frontline agent selects
translation agents to handle user messages.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import httpx
from src.util._constants import (
    HTTP_TIMEOUT_TOTAL,
    HTTP_TIMEOUT_CONNECT,
    HTTP_TIMEOUT_READ,
    HTTP_MAX_KEEPALIVE_CONNECTIONS,
    HTTP_MAX_CONNECTIONS,
    SUPPORTED_LANGUAGES
)

from src.util._client import DeepSeekClient
from src.agents import Agent, ItemHelpers, MessageOutputItem, Runner
from src import set_default_model_client, set_default_model_api
from src.util._pretty_print import pretty_print_result


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


def create_agents() -> tuple[Agent, Agent]:
    """Create translation agents and orchestrator."""
    translation_agents = {
        lang_key.lower(): Agent(
            name=f"{lang_name} Translator",
            instructions=f"Translate English text to {lang_name}",
            handoff_description=f"English to {lang_name} translator",
        )
        for lang_key, lang_name in SUPPORTED_LANGUAGES.items()
    }

    orchestrator_agent = Agent(
        name="Translation Orchestrator",
        instructions=(
            "Coordinate translation requests using provided tools. "
            "Use appropriate translation tools based on requested languages."
        ),
        tools=[
            translation_agents[lang_key.lower()].as_tool(
                tool_name=f"translate_to_{lang_key.lower()}",
                tool_description=f"Translate text to {lang_name}",
            )
            for lang_key, lang_name in SUPPORTED_LANGUAGES.items()
        ],
    )

    return orchestrator_agent


def main() -> str | None:
    """Run the translation service and return the result."""
    try:
        setup_client()
        orchestrator_agent = create_agents()

        msg = input("\n✅ Enter text to translate and target languages: ")
        orchestrator_result = Runner.run_sync(orchestrator_agent, msg)
        
        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"        📝Translation: {text}")

    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Translation service error: {e}", file=sys.stderr)


if __name__ == "__main__":
    if output := main():
        print(output)
