#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent.parent))

from pydantic import BaseModel, TypeAdapter

from src.agents.agent_v1 import AgentV1 as Agent
from src.network.client import setup_client
from src.runners.run import Runner
from src.util.exceptions import AgentExecutionError, ModelError
from src.util.print import validate_json, _format_final_output, pretty_print_result_stream
from src.runners.items import ModelResponse

########################################################
#           Models for Output
########################################################


class OutlineCheckerOutput(BaseModel):
    good_quality: bool
    genre: str


########################################################
#           Agent Creation
########################################################


def create_agents() -> List[Agent]:
    story_outline_agent = Agent(
        name="Agent Outline Generator",
        instructions=(
            "You are a creative story outline generator. "
            "Generate a very short story outline based on the user's input and selected genre. "
            "Keep the outline concise but include key plot points and character development. "
            "Make sure the story fits the selected genre."
        ),
    )

    quality_checker_agent = Agent(
        name="Agent Quality Checker",
        instructions=(
            "You are a story quality analyst. "
            "Read the given story outline and evaluate its quality and genre match. "
            "Focus on coherence, creativity, and whether it fits the selected genre. "
            "Provide your analysis in a clear format with REASONING and RESULT sections. "
            "In the RESULT section, provide a JSON object with 'good_quality' (boolean) and 'genre' (string) fields."
        ),
        output_type=str,
    )

    story_agent = Agent(
        name="Agent Writer",
        instructions=(
            "You are a creative writer. "
            "Write a compelling short story based on the given outline. "
            "Maintain the original plot points while adding engaging details and dialogue. "
            "Ensure the story matches the selected genre."
        ),
        output_type=str,
    )

    return [story_outline_agent, quality_checker_agent, story_agent]


########################################################
#           Private methods
########################################################


def _select_genre() -> str:
    print("\n📚 Available Story Genres:")
    genres = {
        "1": "sci-fi",
        "2": "fantasy",
        "3": "romance",
        "4": "mystery",
        "5": "horror",
        "6": "adventure",
        "7": "solarpunk",
        "8": "lunarpunk",
        "9": "cyberpunk",
        "10": "dystopian",
        "11": "post-apocalyptic",
        "12": "steampunk",
        "13": "urban fantasy",
        "14": "paranormal",
    }

    for num, genre in genres.items():
        print(f"   {num}. {genre.capitalize()}")

    while True:
        genre_choice = input("\n❓ Select a genre number (1-14): ").strip()
        if genre_choice in genres:
            return genres[genre_choice]
        print("❌ Invalid choice. Please select a number between 1 and 14.")


def _extract_and_validate_json(result_text: str) -> OutlineCheckerOutput | None:
    """Extract and validate JSON from the result text."""
    try:
        result_text = result_text.strip()
        start, end = result_text.find("{"), result_text.rfind("}")
        if start == -1 or end == -1:
            print("\n❌ Could not find valid JSON in the response.")
            return None
            
        json_text = result_text[start : end + 1].strip()
        return validate_json(json_text, TypeAdapter(OutlineCheckerOutput))
        
    except ModelError as e:
        print(f"\n❌ Failed to validate checker output: {e}")
        print(f"Raw text: {json_text}")
        return None


def _validate_checker_output(result_text: str, selected_genre: str) -> bool:
    """Validate the checker output and ensure it matches the selected genre."""
    checker_output = _extract_and_validate_json(result_text)
    if checker_output is None:
        return False
        
    if not checker_output.good_quality:
        print("\n❌ Story outline is not of good quality.")
        return False
        
    if checker_output.genre.lower() != selected_genre.lower():
        print(f"\n❌ Story is not a {selected_genre} story.")
        return False
        
    return True


########################################################
#           Agent Runner                               #
########################################################


def run_agents() -> str | None:
    try:
        setup_client()
        agents = create_agents()
        story_outline_agent, quality_checker_agent, story_agent = agents

        selected_genre = _select_genre()
        outline_result = Runner.run_sync(
            story_outline_agent, f"Generate a creative and engaging {selected_genre} story."
        )
        print("\n📝 Generated Outline:")
        print(pretty_print_result_stream(outline_result.final_output))

        checker_result = Runner.run_sync(quality_checker_agent, outline_result.final_output)
        print("\n🔍 Quality Analysis:")
        print(pretty_print_result_stream(checker_result.final_output))

        _, result_text = _format_final_output(
            ModelResponse(
                output=[{"text": str(checker_result.final_output)}],
                usage=None,
                referenceable_id=None,
            )
        )

        if _validate_checker_output(result_text, selected_genre):
            story_result = Runner.run_sync(story_agent, outline_result.final_output)
            print("\n📖 Final Story:")
            print(pretty_print_result_stream(story_result.final_output))
            return story_result.final_output

    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    run_agents()
