# agents "friend with benefits": an example of an agent with orbs

<br>

this example demonstrates a multi-agent system where three agents work together in a chain to create an entertaining text transformation. 

each agent has a specific role and passes its output to the next agent in the chain.

<br>

---

### how it works

1. **first agent**: generates a random number between 3 and 15
2. **second agent**: creates a sentence about the cypherpunk world with exactly that many words
3. **third agent**: modifies the sentence by replacing one word with "love" in an entertaining way

<br>

this example showcases:
- agent chaining using the `orbs` system
- message filtering between agents
- sword integration with agents
- asynchronous agent execution

<br>

---

### running the example

<br>

to run this example, use:

```shell
make friends-with-benefit
```

<br>

here's the implementation:

```python
@function_sword
def random_number() -> int:
    return random.randint(3, 15)


def orbs_message_filter(orbs_message_data: orbsinputdata) -> orbsinputdata:
    """filter the message history to remove swords and keep only relevant history."""
    orbs_message_data = orbsinputfilter.remove_all_swords(orbs_message_data)
    history = (
        tuple(orbs_message_data.input_history[2:])
        if isinstance(orbs_message_data.input_history, tuple)
        else orbs_message_data.input_history
    )
    return orbsinputdata(
        input_history=history,
        pre_orbs_items=tuple(orbs_message_data.pre_orbs_items),
        new_items=tuple(orbs_message_data.new_items),
    )


third_agent = agent(
    name="assistant three",
    instructions="replace one word in the sentence received from agent two with 'love' in a way that makes sense or is entertaining.",
    orbs_description="replace one word in the input sentence with the word 'love'.",
)

second_agent = agent(
    name="agent two",
    instructions="create a sentence about the cypherpunk world with number of words exactly equal to the input number from agent one.",
    orbs=[orbs(third_agent, input_filter=orbs_message_filter)],
    orbs_description="create sentences about the cypherpunk world.",
)

first_agent = agent(
    name="agent one",
    instructions="generate a random between 3 and 15.",
    swords=[random_number],
)


async def run_agent_chain(agent: agent, input_messages: sequence[dict]) -> none:
    result = await runner.run(agent, input=input_messages)
    print(pretty_print_result_stats(result))
    print(pretty_print_result(result, show_reasoning=false))
    return result


async def run_agent():
    try:
        setup_client()

        result = await run_agent_chain(
            first_agent, [{"content": "generate a random number between 3 and 15.", "role": "user"}]
        )

        result = await run_agent_chain(
            second_agent,
            [item.to_input_item() for item in result.new_items]
            + [
                {
                    "content": "create a sentence about the cypherpunk world with the specified word count.",
                    "role": "user",
                }
            ],
        )

        await run_agent_chain(
            third_agent,
            [item.to_input_item() for item in result.new_items]
            + [
                {
                    "content": "replace one word with 'love' in a way that makes sense or is entertaining.",
                    "role": "user",
                }
            ],
        )

    except exception as e:
        raise agentexecutionerror(e) from e


if __name__ == "__main__":
    asyncio.run(run_agent())
```

<br>

the output will be something like this:

```
poetry run python examples/agents/friend_with_benefits.py

✅ Successfully received model response...

✅ RunResult Performance:
  
  👾 Agent Info:
        Last Agent → Agent One
  
  📊 Statistics:
        Items     → 1
        Responses → 1
        Input Shield  → 0
        Output Shield → 0
  
  🦾 Configuration:
        Streaming → ❌ Disabled
        Swords     → Available (1 swords)


✅ RESULT:

The above code will output a random integer **between 3 and 15**, including both 3 and 15. For instance, it might generate: 7


✅ Successfully received model response...

✅ RunResult Performance:
  
  👾 Agent Info:
        Last Agent → Agent Two
  
  📊 Statistics:
        Items     → 1
        Responses → 1
        Input Shield  → 0
        Output Shield → 0
  
  🦾 Configuration:
        Streaming → ❌ Disabled
        Swords     → None


✅ RESULT:

Neon glow lights up derelict streets in a gritty, dark ambiance where neon clashes with the silhouette of night.


✅ Successfully received model response...

✅ RunResult Performance:
  
  👾 Agent Info:
        Last Agent → Assistant Three
  
  📊 Statistics:
        Items     → 1
        Responses → 1
        Input Shield  → 0
        Output Shield → 0
  
  🦾 Configuration:
        Streaming → ❌ Disabled
        Swords     → None


✅ RESULT:

Neon glow lights up derelict streets in a gritty, dark ambiance where neon lovingly clashes with the silhouette of night.
```
