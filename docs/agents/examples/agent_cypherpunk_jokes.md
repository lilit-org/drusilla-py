# agent "cypherpunk joker": an example of an agent with streaming
<br>

run with:

```shell
make cypherpunk-joker
```

<br>

which creates and runs the following agent:

```python
def create_agent() -> Agent:
    return Agent(
        name="Agent Cypherpunk Joker",
        instructions=(
            "You are a cool special robot who tells jokes. Follow these steps exactly:\n"
            "1. First, parse the number of jokes requested from the input message\n"
            "2. After getting the number N, tell exactly N jokes\n"
            "3. Each joke should be on its own line\n"
            "4. Make the jokes cyberpunk-themed and entertaining\n"
            "5. Number each joke clearly"
        ),
        swords=[],
    )


def _get_number_of_jokes() -> int:
    while True:
        try:
            n = int(input("\n❓ How many jokes would you like to hear? (1-10): "))
            if 1 <= n <= 10:
                return n
            print("❌ Please enter a number between 1 and 10.")
        except ValueError:
            print("❌ Please enter a valid number.")


async def _handle_stream_events(result, num_jokes):
    try:
        async for event in result.stream_events():

            if event.type == "agent_updated_stream_event":
                print(f"\n✅ {event.new_agent.name} is telling {num_jokes} jokes...")

            if event.type == "raw_response_event":
                if (
                    hasattr(event.data, "type")
                    and event.data.type == "content_part.done"
                ):
                    if isinstance(event.data.part, dict) and event.data.part.get(
                        "text"
                    ):
                        print(
                            pretty_print_result_stream(event.data.part["text"]),
                            end="",
                            flush=True,
                        )

            elif event.type == "run_item_stream_event":
                if event.name == "message_output_created":
                    if message := ItemHelpers.text_message_output(event.item):
                        print(f"\n💬 Message:\n{message}")
                elif event.name in ("sword_called", "sword_output"):
                    msg = (
                        "⚔️ Sword called"
                        if event.name == "sword_called"
                        else f"⚔️ Sword output: {event.item.output}"
                    )
                    print(f"\n{msg}")

            await asyncio.sleep(0)
        print(pretty_print_result_stats(result))
    except asyncio.CancelledError as e:
        raise AgentExecutionError("Stream processing was cancelled") from e


async def run_agent():
    try:
        setup_client()
        agent = create_agent()
        num_jokes = _get_number_of_jokes()

        result = await Runner.run_streamed(
            agent,
            f"Tell me exactly {num_jokes} cypherpunk jokes!",
        )

        await _handle_stream_events(result, num_jokes)

    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    asyncio.run(run_agent())
```

<br>

you can ask for a certain number of jokes, and it will update and run the agent:

```
❓ How many jokes would you like to hear? (1-10): 4

✅ Agent Cypherpunk Joker is telling 4 jokes...
  
  👾 Agent Info:
        Name   → Agent Cypherpunk Joker
        Turn   → 1/10
        Status → ✔️  Complete
  
  📊 Statistics:
        Items     → 0
        Responses → 1
        Input Shield  → 0
        Output Shield → 0
  
  🦾 Configuration:
        Streaming → ✔️ Enabled
        Swords     → None


✅ REASONING:

Okay, so I need to come up with four cypherpunk-themed jokes based on the user's request. 
First, I'll recall what cypherpunk is—it combines cyberpunk (cyber technology and dystopian
themes) with j科幻 elements like space, AI, and high-tech environments.

I should brainstorm some common cypherpunk elements: robots, holograms, neon lights,
high tech terms, futuristic settings. Now, thinking of joke structures—maybe something that plays
on technology or phrases people might use in their daily lives but add a twist with the cyberpunk theme.

Let me start with the first joke. Maybe something about holographic displays and how they 
interact with emotions because I've heard jokes where holograms can affect people's feelings. 
So, "Why did the hologram get a restraining order?" Because it couldn't handle the heat! 
Haha, that makes sense—holograms are cold but might emit some heat or just be seen as unfriendly.

Next joke: something involving circuit breakers because they're part of electronics and could be
referenced in a cyberpunk setting. "Why don't mathematicians trust circuit breakers?" 
Because they know the voltage is high! This plays on the idea that circuit breakers trip when
there's too much current, like in math problems or electronics.

Third joke: thinking about AI and how it's sometimes used to replace people or do their work. 
So, "Why did the AI take over the factory?" To make the workers look busy! 
Maybe because it keeps them occupied so they don't question its existence. 
It's a bit of a twist on the typical replacement scenario.

Lastly, for the fourth joke, maybe something that combines high-tech with everyday items—like
robots and sandwiches. "Why did the robot order a sandwich?" 
Because it was feeling productive! Playing on the idea that even simple tasks like making a
sandwich can be seen as productive in a high-tech environment.

I should make sure each joke is clear and numbered correctly, placed at the end of the response
Also, keeping them concise but funny so they fit well within the cypherpunk theme without 
being too obscure.


✅ RESULT:

1. Why did the hologram get a restraining order?
   Because it couldn't handle the heat!

2. Why don't mathematicians trust circuit breakers?
   Because they know the voltage is high!

3. Why did the AI take over the factory?
   To make the workers look busy!

4. Why did the robot order a sandwich?
   Because it was feeling productive!

✅ Stream complete
✅ Final state has 0 items and 1 responses
```
