# agents

<br>

## tl; dr

<br>

- agents are the main objects implementing workflow loops in the drusilla framework
- agents are made of generic context objects, which are created to specify the agent's behavior and gear, and passed to [`Runner.run()`](https://github.com/lilit-org/drusilla-py/blob/main/docs/primitives/runners.md#overview-of-the-runner-class)
- the agent's underlying model is 
- agents can be created with:
    - configurable instructions and model settings, set by the [model's settings](../primitives/models.md)
    - swords for performing specific actions
    - shields that allow input and output validation
    - orbs that are sub-agents the agent can delegate to
    - charms for monitoring and modifying an agent's behavior and lifecycle
- `output_types` are either `str` or customized objects (i.e., pydantic objects, dataclasses, lists, `TypedDict`)

<br>

---

## content

<br>

- [the `AgentV1` class](#the-agentv1-class)
    - [creating a simple agent](#creating-a-simple-agent)
    - [agent as a sword](#agent-as-a-sword)
- [agent cypherpunk love, a hello world example](#agent-cypherpunk-love-a-hello-world-example)
- [other examples](#other-examples)

<br>

---

## the `AgentV1` class

<br>

the [AgentV1 class](../../src/agents/agent_v1.py) is the first version of our agents:

<br>

```python
@dataclass
class AgentV1(Generic[TContext]):
    """AI agent with swords, shields, orbs, and charms."""

    name: str
    instructions: (
        str
        | Callable[
            [RunContextWrapper[TContext], AgentV1[TContext]],
            MaybeAwaitable[str],
        ]
        | None
    ) = None

    orbs_description: str | None = None
    orbs: list[AgentV1[Any] | Orbs[TContext]] = field(default_factory=list)
    model: str | Model = field(default="")
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    swords: list[Sword] = field(default_factory=list)
    input_shields: list[InputShield[TContext]] = field(default_factory=list)
    output_shields: list[OutputShield[TContext]] = field(default_factory=list)
    output_type: type[Any] | None = None
    charms: AgentCharms[TContext] | None = None
    sword_use_behavior: (
        Literal["run_llm_again", "stop_on_first_sword"]
        | dict[Literal["stop_at_sword_names"], list[str]]
    ) = "run_llm_again"

    def clone(self, **kwargs: Any) -> AgentV1[TContext]:
        """Create a copy of the agent with optional field updates."""
        return replace(self, **kwargs)

    def as_sword(
        self,
        sword_name: str | None = None,
        sword_description: str | None = None,
        custom_output_extractor: Callable[[Any], str] | None = None,
    ) -> Sword:
        """Converts agent to sword for other agents."""
        return _create_agent_sword(self, sword_name, sword_description, custom_output_extractor)

    async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
        """Get the system prompt for the agent."""
        if self.instructions is None:
            return None

        if isinstance(self.instructions, str):
            return self.instructions

        if not callable(self.instructions):
            raise set_error(
                AgentExecutionError,
                error_messages.AGENT_EXEC_ERROR,
                error=f"Invalid instructions type: {type(self.instructions)}"
            )

        try:
            result = self.instructions(run_context, self)
            if inspect.iscoroutinefunction(self.instructions):
                return await cast(Awaitable[str], result)
            return cast(str, result)
        except Exception as e:
            raise set_error(
                AgentExecutionError,
                error_messages.AGENT_EXEC_ERROR,
                error=str(e)
            ) from e


@dataclass(init=False)
class AgentV1OutputSchema:
    """Schema for validating and parsing LLM output into specified types."""

    output_type: type[Any]
    strict_json_schema: bool
    _type_adapter: TypeAdapter[Any]
    _is_plain_text: bool

    def __init__(self, output_type: type[Any], strict_json_schema: bool = True):
        """
        Args:
            output_type: Type to validate against
            strict_json_schema: Enable strict validation (recommended)
        """
        self.output_type = output_type
        self.strict_json_schema = strict_json_schema
        self._is_plain_text = output_type is None or output_type is str
        self._type_adapter = get_type_adapter(output_type)

    def is_plain_text(self) -> bool:
        return self._is_plain_text

    def json_schema(self) -> dict[str, Any]:
        if self.is_plain_text():
            raise set_error(
                UsageError,
                error_messages.USAGE_ERROR,
                error="No JSON schema for plain text output"
            )
        schema = self._type_adapter.json_schema()
        if self.strict_json_schema:
            schema = ensure_strict_json_schema(schema)
        return schema

    def validate_json(self, json_str: str, partial: bool = False) -> Any:
        return validate_json(json_str, self._type_adapter, partial)

    def output_type_name(self) -> str:
        return type_to_str(self.output_type)
```

<br>

---

### creating a simple agent

<br>

```python
agent = Agent(
        name="Agent Dr. Love",
        instructions="You are a cool special robot who loves"
    )

result = Runner.run_sync(
        agent,
        "Write a haiku about love in the cypherpunk world."
    )
```

<br>


---

### agent as a sword

<br>

agents can be used standalone or converted into swords for other agents, enabling
composability.

```python
def _create_agent_sword(
    agent: Agent[TContext],
    sword_name: str | None,
    sword_description: str | None,
    custom_output_extractor: Callable[[Any], str] | None = None,
) -> Sword:
    @function_sword(
        name_override=sword_name or transform_string_function_style(agent.name),
        description_override=sword_description or "",
    )
    async def _agent_sword(ctx: RunContextWrapper[TContext], input: str) -> Any:
        from ..runners.run import Runner

        result = await Runner.run(
            starting_agent=agent, input=input, context=ctx, max_turns=MAX_TURNS
        )
        if custom_output_extractor:
            return custom_output_extractor(result.final_output)
        return result.final_output

    return _agent_sword
```

<br>

---

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
* [agent friend with benefits](examples/agent_friend_with_benefits.md): an example of an agent with orbs
* [agent cyphepunk jokes](examples/agent_cypherpunk_jokes.md): an example of an agent with streaming
* [agent cypherpunk story](examples/agent_cypherpunk_storyteller.md): an example of an agent with orbs
