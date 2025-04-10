# agent "cypherpunk storyteller": an example of a multi-agent story generation system
<br>

run with:

```shell
make cypherpunk-storyteller
```

<br>

which creates and runs the following agents:

```python
def create_agents() -> List[Agent]:
    story_outline_agent = Agent(
        name="Story Outline Generator",
        instructions=(
            "You are a creative story outline generator. "
            "Generate a very short story outline based on the user's input and selected genre. "
            "Keep the outline concise but include key plot points and character development. "
            "Make sure the story fits the selected genre."
        ),
    )

    outline_checker_agent = Agent(
        name="Outline Quality Checker",
        instructions=(
            "You are a story quality analyst. "
            "Read the given story outline and evaluate its quality and genre match. "
            "Focus on coherence, creativity, and whether it fits the selected genre. "
            "Provide your analysis in a clear, structured format."
        ),
        output_type=str,
    )

    json_parser_agent = Agent(
        name="JSON Parser",
        instructions=(
            "You are a JSON parser agent. "
            "Convert the given quality analysis into a valid JSON object. "
            "The JSON must have two fields: 'good_quality' (boolean) and 'genre' (string). "
            "CRITICAL: Your response must be ONLY a raw JSON object with no additional text, tags, or formatting. "
            "DO NOT include any thinking tags, process text, or markdown. "
            "DO NOT explain your reasoning. "
            "DO NOT include any newlines before or after the JSON. "
            "The response must be a single line starting with { and ending with }. "
            "Example of valid output: "
            "{\"good_quality\": true, \"genre\": \"sci-fi\"}"
            "IMPORTANT: Do not include <think> or </think> tags or any other text. Only output the JSON object."
        ),
        output_type=OutlineCheckerOutput,
    )

    story_agent = Agent(
        name="Story Writer",
        instructions=(
            "You are a creative writer. "
            "Write a compelling short story based on the given outline. "
            "Maintain the original plot points while adding engaging details and dialogue. "
            "Ensure the story matches the selected genre."
        ),
        output_type=str,
    )

    return [story_outline_agent, outline_checker_agent, json_parser_agent, story_agent]
```

<br>

The system uses four agents working together:

1. **Story Outline Generator**: Creates a concise story outline based on the selected genre
2. **Outline Quality Checker**: Analyzes the outline's quality and genre match
3. **JSON Parser**: Converts the quality analysis into a structured JSON format
4. **Story Writer**: Expands the outline into a full story

The process works as follows:

1. User selects a genre from the available options:
   - Sci-fi
   - Fantasy
   - Romance
   - Mystery
   - Horror
   - Adventure
   - Solarpunk
   - Lunarpunk
   - Cyberpunk
   - Dystopian
   - Post-apocalyptic
   - Steampunk
   - Urban fantasy
   - Paranormal

2. The Story Outline Generator creates a brief outline

3. The Outline Quality Checker analyzes:
   - Story quality
   - Genre match
   - Provides a detailed analysis

4. The JSON Parser converts the analysis into a structured format:
   - `good_quality`: boolean indicating if the story meets quality standards
   - `genre`: string confirming the story's genre

5. If the outline passes validation, the Story Writer creates the full story

Example interaction:

```
📚 Available Story Genres:
   1. Sci-fi
   2. Fantasy
   3. Romance
   4. Mystery
   5. Horror
   6. Adventure
   7. Solarpunk
   8. Lunarpunk
   9. Cyberpunk
   10. Dystopian
   11. Post-apocalyptic
   12. Steampunk
   13. Urban fantasy
   14. Paranormal

❓ Select a genre number (1-14): 1

📝 Generated Outline:
[Story outline appears here]

🔍 Quality Analysis:
[Detailed quality analysis appears here]

📊 Quality Check Results:
{"good_quality": true, "genre": "sci-fi"}

📖 Final Story:
[Full story appears here]
```

The system ensures that:
- Stories match the selected genre
- Outlines are of good quality
- Quality analysis is properly structured
- The final story maintains the original plot while adding engaging details
- All output is properly formatted and validated 