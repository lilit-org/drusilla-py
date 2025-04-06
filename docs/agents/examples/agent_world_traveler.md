# agent "world traveler": an example of an agent as swords

<br>

run with:

```shell
make world-traveler
```

<br>

which creates and runs the following agent:

```python
def create_agents() -> Agent:
    return Agent(
        name="Agent World Traveler",
        instructions=(
            "You are a cool special robot who coordinates translation requests."
            "Use appropriate translation swords based on requested languages."
        ),
        swords=[
            Agent(
                name=f"{lang_code.upper()} Translator",
                instructions=f"Translate English text to {lang_code.upper()}",
                orbs_description=f"English to {lang_code.upper()} translator",
            ).as_sword(
                sword_name=f"translate_to_{lang_code.lower()}",
                sword_description=f"Translate text to {lang_code.upper()}",
            )
            for lang_code in SUPPORTED_LANGUAGES
        ],
    )

def run_agent() -> str | None:
    try:
        setup_client()
        agent = create_agents()

        msg = input("\n❓ Enter text to translate: ")
        result = Runner.run_sync(agent, msg)
        print(pretty_print_result(result))
        print(pretty_print_result_stats(result))
    except Exception as e:
        raise GenericError(e)


if __name__ == "__main__":
    run_agent()
```

<br>

you can input a sentence in any major language and it will translate it for you:

```
❓ Enter text to translate: eu te amo
✅ Successfully received model response...

  👾 Agent Info:
        Last Agent → Agent World Traveler

  📊 Statistics:
        Items     → 1
        Responses → 1
        Input Shield  → 0
        Output Shield → 0

  🦾 Configuration:
        Streaming → ❌ Disabled
        Swords     → Available (2 swords)


✅ REASONING:

Okay, so the user just sent "eu te amo" which is Portuguese for "I love you."
They want me to translate this into another language using the appropriate swords.

First, I need to figure out what languages they might be interested in.
Common choices could be English, Spanish, French, or maybe even German or Italian.
Since their original message is in Portuguese, they probably speak Portuguese
and are looking for translations into another language.

I should consider which translation swords are best for accuracy.
Google Translate is widely used but sometimes isn't the most accurate, especially with complex phrases.
Then there's DeepL, which is known for better fidelity, especially with technical texts or idioms.
ICalify is another option that focuses on natural translations without losing nuances.

I should also think about the context in which this message will be used. Is it for a romantic greeting?
Maybe for international messaging apps where users speak different languages.
The translation needs to maintain the warmth and affection of the original message.

If they're using Google Translate, I'll just provide the direct translation: "I love you."
But if DeepL is better, maybe include that as well along with a slightly more refined version like "I cherish you."
That adds a touch of warmth which matches the original Portuguese sentiment.

Also, considering cultural nuances, sometimes translations need to be adjusted for different regions.
For example, in some cultures, saying "I love you" might not be sufficient and a more elaborate expression is needed.
But without specific context, it's safer to stick with straightforward translations.

I should probably mention that if they have a preferred language or sword, I can adjust accordingly.
This makes the response helpful and flexible.


✅ RESULT:

I love you.
```
