# agents

<br>

## tl; dr

<br>

agents are the main object implementing workflows in the drusilla framework.

<br>

---

## contents

<br>

- [the `AgentV1` class](#the-agentv1-class)
    - [creating a simple agent](#creating-a-simple-agent)
    - [the agent's gear](#the-agents-gear)
    - [agent as a sword](#agent-as-a-sword)
- [available examples](#available-examples)

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
    orbs: list[Agent[Any] | Orbs[TContext]] = field(default_factory=list)
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
```

<br>

and the following output schema for validating and parsing output into specified types:

<br>

```python
@dataclass(init=False)
class AgentV1OutputSchema:

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
            raise UsageError("No JSON schema for plain text output")
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

### the agent's gear

<br>

agents can be created with:

- configurable instructions and model settings
- swords for performing specific actions
- shields for safety and validation of inputs and outputs
- orbs to other agents or handlers
- charms for monitoring and modifying an agent's behavior

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
            starting_agent=agent, input=input, context=ctx, max_turns=config.MAX_TURNS
        )
        if custom_output_extractor:
            return custom_output_extractor(result.final_output)
        return result.final_output

    return _agent_sword
```

<br>

---

### available examples

<b>

check out [examples/agents](../../examples/agents/) to see learn from the examples we provide.
