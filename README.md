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

create a `.env` file in your project root with your deepseek api endpoint:

```shell
BASE_URL = "http://localhost:11434"           
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

```
Encrypted hearts pulse,
Digital whispers unite —
Secret love in code.
```
