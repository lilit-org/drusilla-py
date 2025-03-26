#!/usr/bin/env python3

"""
This script demonstrates the agents-as-tools pattern where a frontline agent selects
translation agents to handle user messages.
"""

import sys
import httpx
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.util._client import DeepSeekClient
from src.agents import Agent, ItemHelpers, MessageOutputItem, Runner
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


def create_agents() -> tuple[Agent, Agent]:
    """Create translation agents and orchestrator."""
    spanish_agent = Agent(
        name="Spanish Translator",
        instructions="Translate English text to Spanish",
        handoff_description="English to Spanish translator",
    )

    portuguese_agent = Agent(
        name="Portuguese Translator",
        instructions="Translate English text to Portuguese",
        handoff_description="English to Portuguese translator",
    )

    french_agent = Agent(
        name="French Translator",
        instructions="Translate English text to French",
        handoff_description="English to French translator",
    )

    italian_agent = Agent(
        name="Italian Translator",
        instructions="Translate English text to Italian",
        handoff_description="English to Italian translator",
    )

    japanese_agent = Agent(
        name="Japanese Translator",
        instructions="Translate English text to Japanese",
        handoff_description="English to Japanese translator",
    )

    orchestrator_agent = Agent(
        name="Translation Orchestrator",
        instructions=(
            "Coordinate translation requests using provided tools. "
            "Use appropriate translation tools based on requested languages."
        ),
        tools=[
            portuguese_agent.as_tool(
                tool_name="translate_to_portuguese",
                tool_description="Translate text to Portuguese",
            ),
            spanish_agent.as_tool(
                tool_name="translate_to_spanish",
                tool_description="Translate text to Spanish",
            ),
            french_agent.as_tool(
                tool_name="translate_to_french",
                tool_description="Translate text to French",
            ),
            italian_agent.as_tool(
                tool_name="translate_to_italian",
                tool_description="Translate text to Italian",
            ),
            japanese_agent.as_tool(
                tool_name="translate_to_japanese",
                tool_description="Translate text to Japanese",
            ),
        ],
    )

    synthesizer_agent = Agent(
        name="World Traveler",
        instructions="Review and combine translations into final response.",
    )

    return orchestrator_agent, synthesizer_agent


def main() -> str | None:
    """Run the translation service and return the result."""
    try:
        setup_client()
        orchestrator_agent, synthesizer_agent = create_agents()
        
        msg = input("\n👾 Enter text to translate and target languages: ")
        
        orchestrator_result = Runner.run_sync(orchestrator_agent, msg)
        
        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"  - Translation: {text}")
        
        synthesizer_result = Runner.run_sync(
            synthesizer_agent, orchestrator_result.to_input_list()
        )
        
        return pretty_print_result(synthesizer_result)
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Translation service error: {e}", file=sys.stderr)
    return None


if __name__ == "__main__":
    if output := main():
        print(output)
