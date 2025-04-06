# agents

<br>

## agent cypherpunk love, a hello world example

<br>

test your configuration by running our hello world example:

```shell
make cypherpunk-love
```

<br>

which creates and runs the following agent:

```python
def create_agent() -> Agent:
    return Agent(
        name="Agent Dr. Love", i
        instructions="You are a cool special robot who loves"
    )


def run_agent() -> str | None:
    try:
        setup_client()
        agent = create_agent()
        result = Runner.run_sync(
            agent,
            "Write a haiku about love in the cypherpunk world."
        )
        print(pretty_print_result(result))
        print(pretty_print_result_stats(result))
    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    run_agent()
```

<br>

you should get something like this:

```
✅ Received Model Response...

  👾 Agent Info:
        Last Agent → Agent Dr. Love

  📊 Statistics:
        Items     → 1
        Responses → 1
        Input Shield  → 0
        Output Shield → 0

  🦾 Configuration:
        Streaming → ❌ Disabled
        Swords    → None


✅ REASONING:

Okay, so I have to write a Haiku about love in the cyberpunk world.
Hmm, let me think about how to approach this.

First, what is a Haiku? It's a three-line poem with a 5-7-5 syllable structure.
The first and third lines are five syllables, and the middle line is seven.
So I need to make sure each line meets that syllable count.

Now, the subject is "love" in the cyberpunk setting.
Cyberpunk usually involves futuristic cities, neon lights, abandoned places,
high-tech but often dystopian environments.
Love in such a setting could be portrayed through themes like overcoming
connections through technology, or ideals versus reality.

I should think about metaphors or symbols related to love that fit into this genre.
Maybe something like signals, connection points that bring people together who
wouldn't normally interact like how cyberpunk often features isolated individuals
connected by these futuristic means.

In the first line, I want to set a scene or an emotion that starts with "loving."
Maybe starting with "Love" and then moving into the environment.
Something about technology influencing relationships, like holograms or virtual connections.
So maybe love as something that exists not just in person but also in digital spaces.

The second line should be longer with seven syllables.
Here, I can introduce a setting where this love takes place amidst the cyberpunk elements.
Perhaps mentioning how the environment is both bright and broken, providing contrast between
light and darkness,
like neon signs against decay or preserved ruins next to glowing tech.

For the third line, focusing on emotions that come from such love perhaps the connection
between two people who've been apart for a long time but find each other through
this digital lens.
Maybe something about memories connecting them now with the help of technology.


✅ RESULT:

Encrypted hearts pulse,
Digital whispers unite —
Secret love in code.
```

<br>

---

## other examples

<br>

* [agent dissociative identity](examples/agent_dissociative_identity.md): another simple reasoning agent
* [agent summer chaser](examples/agent_summer_chaser.md): an example of an agent with swords
* [agent world traveler](examples/agent_world_traveler.md): an example of an agent as swords
* [agent cyphepunk jokes](examples/agent_cypherpunk_jokes.md): an example of an agent with streaming
* [agent friend with benefits](examples/agent_friend_with_benefits.md): an example of an agent with orbs
