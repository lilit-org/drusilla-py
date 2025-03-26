# ⽊ lilit's deepseek-r1 agentic framework 

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/f473ad0c-82e7-40b7-9aea-e4de45c8d360" width="70%" align="center" style="padding:1px;border:1px solid black;"/>
</p>


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

install dependencies

```shell
make install
```

<br>

create a `.env` file in your project root with your deepseek api endpoint and any customization:

```shell
BASE_URL = "http://localhost:11434"      
LOGGING = "DEBUG"     
```

<br>

start [ollama](https://ollama.com/) in another terminal window (after downloading [deepseek-r1](https://ollama.com/library/deepseek-r1)):

```shell
ollama serve
```

<br>

___

## example

<br>

test your configuration by running:

```shell
make sanity-test
```

<br>

which creates and runs the following agent:

```shell
def setup_client() -> DeepSeekClient:
    client = DeepSeekClient(
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=30.0, read=90.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    )
    set_default_model_client(client)
    set_default_model_api("chat_completions")
    return client


def create_agent() -> Agent:
    return Agent(
        name="Agent Mulder",
        instructions="You are a cool special agent robot"
    )


def main() -> Optional[str]:
    try:
        setup_client()
        agent = create_agent()
        
        result = Runner.run_sync(
            agent,
            "Write a haiku about love in the cypherpunk world."
        )
        return result.final_output
    except Exception as e:
        print(f"Error running sanity test: {e}", file=sys.stderr)
```

<br>

you should get something like this:

```shell
✅ Received Model Response...
✅ RunResult:
  
  👾 Agent Info:
        Last Agent : Agent Mulder
  
  📊 Statistics:
        Items     : 1
        Responses : 1
        Input GR  : 0
        Output GR : 0
  
  🦾 Configuration:
        Streaming  : ❌ Disabled
        Tool Mode  : None
        Response   : None

✅ REASONING:

Okay, so I have to write a Haiku about love in the cyberpunk world.
Hmm, let me think about how to approach this.

First, what is a Haiku? It's a three-line poem with a 5-7-5 syllable structure.
The first and third lines are five syllables, and the middle line is seven.
So I need to make sure each line meets that syllable count.

Now, the subject is "love" in the cyberpunk setting. Cyberpunk usually involves futuristic cities, neon lights, abandoned places, high-tech but often dystopian environments.
Love in such a setting could be portrayed through themes like overcominf connections through technology, or ideals versus reality.

I should think about metaphors or symbols related to love that fit into this genre.
Maybe something like signals, connection points that bring people together who wouldn't normally interact ike how cyberpunk often features isolated individuals connected by these futuristic means.

In the first line, I want to set a scene or an emotion that starts with "loving."
Maybe starting with "Love" and then moving into the environment.
Something about technology influencing relationships, like holograms or virtual connections.
So maybe love as something that exists not just in person but also in digital spaces.

The second line should be longer with seven syllables.
Here, I can introduce a setting where this love takes place amidst the cyberpunk elements
Perhaps mentioning how the environment is both bright and broken, providing contrast between light and darkness, like neon signs against decay or preserved ruins next to glowing tech.

For the third line, focusing on emotions that come from such love perhaps the connection between two people who've been apart for a long time but find each other through this digital lens.
Maybe something about memories connecting them now with the help of technology.


✅ RESULT:

Encrypted hearts pulse,
Digital whispers unite —
Secret love in code.
```
