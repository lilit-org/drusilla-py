# ⽊ lilit's deepseek-r1 agentic framework 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/f473ad0c-82e7-40b7-9aea-e4de45c8d360" width="90%" align="center" style="padding:1px;border:1px solid black;"/>
</p>


<br>

---

## overview

<br>

this project was inspired by many open-source frameworks and our own local workflows, and customized for deepseek and for the work we are doing at **[lilit](https://github.com/lilit-org)**.



### primitives

to design multi-agent systems, we utilize the following primitives:

- [agents](src/agents/agent.py): our LLM robots that can be equipped with orbs and shields
- [orbs](src/gear/orbs.py): part of the agent's gear, used to delegate tasks to other agents
- [shields](src/gear/shields.py): part of the agent's gear, used to validate and protect the inputs from agents


<br>

---

## local development

<br>

set up your python environment:

```shell
python3 -m venv venv
source venv/bin/activate
```

<br>

install dependencies:

```shell
make install
```

<br>

create a `.env` file in your project root with your deepseek api endpoint and any customization (or you can leave the default values):

```shell
BASE_URL = "http://localhost:11434"
MODEL = "deepseek-r1"
MAX_TURNS = 10
WRAPPER_DICT_KEY = "response"
MAX_QUEUE_SIZE = 1000
MAX_GUARDRAIL_QUEUE_SIZE = 100
LRU_CACHE_SIZE = 128
LOG_LEVEL = "DEBUG"  
HTTP_TIMEOUT_TOTAL = 120.0
HTTP_TIMEOUT_CONNECT = 30.0
HTTP_TIMEOUT_READ = 90.0
HTTP_MAX_KEEPALIVE_CONNECTIONS = 5
HTTP_MAX_CONNECTIONS = 10

```

<br>

start [ollama](https://ollama.com/) in another terminal window (after downloading [deepseek-r1](https://ollama.com/library/deepseek-r1)):

```shell
ollama serve
```

<br>

___

## agents

<br> 


### agent "cypherpunk love": a hello world

<br>

test your configuration by running:

```shell
make cypherpunk-love
```

<br>

which creates and runs the following agent:

```python
def create_agent() -> Agent:
    return Agent(
        name="Agent Dr. Love",
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
    except Exception as e:
        raise GenericError(e)


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
        Input GR  → 0
        Output GR → 0
  
  🦾 Configuration:
        Streaming → ❌ Disabled
        Tool Mode → None

✅ REASONING:

Okay, so I have to write a Haiku about love in the cyberpunk world.
Hmm, let me think about how to approach this.

First, what is a Haiku? It's a three-line poem with a 5-7-5 syllable structure.
The first and third lines are five syllables, and the middle line is seven.
So I need to make sure each line meets that syllable count.

Now, the subject is "love" in the cyberpunk setting.
Cyberpunk usually involves futuristic cities, neon lights, abandoned places, high-tech but often dystopian environments.
Love in such a setting could be portrayed through themes like overcoming connections through technology, or ideals versus reality.

I should think about metaphors or symbols related to love that fit into this genre.
Maybe something like signals, connection points that bring people together who wouldn't normally
interact like how cyberpunk often features isolated individuals connected by these futuristic means.

In the first line, I want to set a scene or an emotion that starts with "loving."
Maybe starting with "Love" and then moving into the environment.
Something about technology influencing relationships, like holograms or virtual connections.
So maybe love as something that exists not just in person but also in digital spaces.

The second line should be longer with seven syllables.
Here, I can introduce a setting where this love takes place amidst the cyberpunk elements.
Perhaps mentioning how the environment is both bright and broken, providing contrast between light and darkness,
like neon signs against decay or preserved ruins next to glowing tech.

For the third line, focusing on emotions that come from such love perhaps the connection between two people who've
been apart for a long time but find each other through this digital lens.
Maybe something about memories connecting them now with the help of technology.


✅ RESULT:

Encrypted hearts pulse,
Digital whispers unite —
Secret love in code.
```

<br>

---

### agent "world traveler": using tools

<br>

run a first example of a simple multi-agent with:

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
            "Coordinate translation requests using provided tools. "
            "Use appropriate translation tools based on requested languages."
        ),
        tools=[
            Agent(
                name=f"{lang_name} Translator",
                instructions=f"Translate English text to {lang_name}",
                handoff_description=f"English to {lang_name} translator",
            ).as_tool(
                tool_name=f"translate_to_{lang_key.lower()}",
                tool_description=f"Translate text to {lang_name}",
            )
            for lang_key, lang_name in SUPPORTED_LANGUAGES.items()
        ],
    )


def run_agent() -> str | None:
    try:
        setup_client()
        agent = create_agents()

        msg = input("\n❓ Enter text to translate and target languages: ")
        result = Runner.run_sync(agent, msg)
        print(pretty_print_result(result))
    except Exception as e:
        raise GenericError(e)


if __name__ == "__main__":
    run_agent()
```

<br>

you can input a sentence in any major language and it will translate it for you:

```
❓ Enter text to translate and target languages: eu te amo
✅ Successfully received model response...

  👾 Agent Info:
        Last Agent → Agent World Traveler

  📊 Statistics:
        Items     → 1
        Responses → 1
        Input GR  → 0
        Output GR → 0

  🦾 Configuration:
        Streaming → ❌ Disabled
        Tool Mode → None

✅ REASONING:

Okay, so the user just sent "eu te amo" which is Portuguese for "I love you."
They want me to translate this into another language using the appropriate tools.

First, I need to figure out what languages they might be interested in.
Common choices could be English, Spanish, French, or maybe even German or Italian.
Since their original message is in Portuguese, they probably speak Portuguese
and are looking for translations into another language.

I should consider which translation tools are best for accuracy.
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

I should probably mention that if they have a preferred language or tool, I can adjust accordingly.
This makes the response helpful and flexible.


✅ RESULT:

I love you.
```
