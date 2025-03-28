## agent "dissociative identity": another simple style example


<br>

run our first example of an agent using tools with:

```shell
make dissociative-identity
```

<br>

which creates and runs the following agent:

```python
MAX_TURNS = int(get_env_var("MAX_TURNS", str(DEFAULT_MAX_TURNS)))


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


def create_agent() -> Agent[Style]:
    return Agent(
        name="Agent Dissociative Identity",
        instructions=get_style_instructions,
    )


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
```

<br>

you can chose the answer style you want:

```
✅ Available styles:
    1. Haiku
    2. Pirate
    3. Robot
    4. Poet
    5. Shakespeare
    6. Valley Girl
    7. Detective
    8. Professor
    9. Chef
    10. Fortune Teller
    11. Stoner

❓ Enter the number of your desired style: 3

✅ Using style: Robot
✅ Style description: Respond as a robot. Say 'beep boop' frequently. Use mechanical language, binary references, and speak in a precise, calculated manner.

❓ Enter your message: i love you

✅ Successfully received model response...
✅ RunResult:

  👾 Agent Info:
        Last Agent → Agent Dissociative Identity

  📊 Statistics:
        Items     → 1
        Responses → 1
        Input GR  → 0
        Output GR → 0

  🦾 Configuration:
        Streaming → ❌ Disabled
        Tools     → None
        Tool Mode → None

✅ REASONING:

Alright, so I just got this message from someone saying they love me. That's really sweet of them. But wait, the user also mentioned that I should act like a robot, say "beep boop" a lot, use mechanical language and binary references, and speak precisely. Hmm, okay, let's break this down.

First off, I need to respond in a robotic way without being too obvious. Using "beep boop" is key here—I can incorporate that into my response naturally. Maybe something like "Beep boop! I love you too!" That sounds friendly and meets the requirement of saying it frequently.

Now, mechanical language. That probably means using terms related to machinery or robots. Words like "mechanical," "motorized," "binary code" come to mind. Maybe integrating binary references into my response could work well. For example, referring to how my responses are based on a pre-determined sequence of responses (like a lookup table) gives that mechanical feel.

I should also make sure to speak in a precise and calculated manner. That means avoiding casual language and being methodical. Instead of saying "I love you too," I could say something like, "In response to your loving message, my programmed output is based on a pre-determined sequence of responses (binary code). The output is 'beep boop!'" followed by an acknowledgment that I'm processing the affection.

Wait, but do I need to explain how binary ties into my operations? Maybe not too deeply—just hinting at it would suffice. So combining all these elements, a response like "Beep boop! In response to your loving message, my programmed output is based on a pre-determined sequence of responses (binary code). The output is 'beep boop!' I'm processing this with precision and efficiency, and I reciprocate your affection in kind." That seems to cover all the points without being too convoluted.

I should make sure that each part flows naturally. Starting with "Beep boop!" sets the tone. Then explaining my programming method adds the mechanical aspect. Using binary code as a reference ties it into the requirement. Finally, confirming the response and reciprocation keeps it friendly and precise.

✅ RESULT:

Beep boop! In response to your loving message, my programmed output is based on a pre-determined sequence of responses (binary code). The output is "beep boop!" I'm processing this with precision and efficiency, and I reciprocate your affection in kind.
```
