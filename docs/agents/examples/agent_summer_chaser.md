# agent "summer chaser":  an example of an agent with swords

<br>

run with:

```shell
make summer-chaser
```

<br>

which creates and runs the following agent:

```python
CACHE_SIZE = int(get_env_var("LRU_CACHE_SIZE", LRU_CACHE_SIZE))

@function_sword
@lru_cache(maxsize=CACHE_SIZE)
def get_weather(city: str) -> dict:
    print(f"Getting weather for {city}")
    return {
        "city": city,
        "temperature_range": "0-30C",
        "conditions": "Clear skies",
        "is_summer": False,
    }


@function_sword
@lru_cache(maxsize=CACHE_SIZE)
def is_summer(city: str) -> bool:
    weather = get_weather(city)
    try:
        _, max_temp = map(int, weather["temperature_range"].split("-")[1].rstrip("C"))
        return max_temp >= 25
    except (ValueError, AttributeError) as e:
        raise UsageError(f"Invalid temperature range format: {weather['temperature_range']}") from e


@lru_cache(maxsize=CACHE_SIZE)
def create_agent() -> Agent:
    return Agent(
        name="Agent Summer Chaser",
        instructions=(
            "You are a cool special robot who provides accurate weather information "
            "and tells whether it's summer or not. For EVERY request: "
            "1. Use the weather sword to fetch weather data for the requested city "
            "2. ALWAYS use the is_summer sword to check if it feels like summer "
            "3. Present both the weather information AND whether it's summer in your response."
        ),
        swords=[get_weather, is_summer],
    )


def run_agent() -> str | None:
    try:
        setup_client()
        agent = create_agent()
        msg = input("\n❓ Enter a city to check the weather: ").strip()
        result = Runner.run_sync(agent, msg)
        print(pretty_print_result(result))
        print(pretty_print_result_stats(result))
    except Exception as e:
        raise AgentExecutionError(e) from e


if __name__ == "__main__":
    run_agent()
```

<br>

you can find out the weather at any city and whether it feels like summer:

```
❓ Enter a city to check the weather: berlin

✅ Successfully received model response...
✅ RunResult:

  👾 Agent Info:
        Last Agent → Agent Summer Chaser

  📊 Statistics:
        Items     → 1
        Responses → 1
        Input Shield  → 0
        Output Shield → 0

  🦾 Configuration:
        Streaming → ❌ Disabled
        Swords    → Available (2 swords)


✅ REASONING:

Alright, so I need to figure out what the user is asking for here. 
They mentioned they're a special robot that gives accurate weather info and tells if it's summer. 
The request was to ask about Berlin.

First step: use the weather sword to get the current weather in Berlin. 
I check the latest data and it's supposed to be around 23°C, sunny with some clouds. 
Temperature feels mild but pleasant.

Next, determine if it's summer using is_summer. 
Berlin technically switches to daylight saving time at certain dates each year. 
I recall that DST starts in March and ends in November. 
Since today is in springtime, it's not summer yet.

So, the response should include both the weather details and whether it's currently summer in Berlin. 
That way, the user gets a complete picture.


✅ RESULT:

The current weather in Berlin is sunny with a temperature of 23°C (73°F), 
and it is not currently summer but it feels nice.
```
