# agents

<br>

## overview

<br>

the [agent class](../../src/agents/agent.py) is the main dataclass to implement an agent, providing a flexible framework for creating decentralized AI agents.

```python
@dataclass
class Agent(Generic[TContext]):
    name: str
    instructions: (
        str
        | Callable[
            [RunContextWrapper[TContext], Agent[TContext]],
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
    hooks: AgentHooks[TContext] | None = None
    sword_use_behavior: (
        Literal["run_llm_again", "stop_on_first_sword"]
        | dict[Literal["stop_at_sword_names"], list[str]]
        | SwordsToFinalOutputFunction
    ) = "run_llm_again"

    def as_sword(
        sword_name: str | None,
        sword_description: str | None,
        custom_output_extractor: Callable[[Any], str] | None = None,
    ) -> Sword:
        return _create_agent_sword(self, sword_name, sword_description, custom_output_extractor)

    async def get_system_prompt(self, run_context: RunContextWrapper[TContext]) -> str | None:
        """Get the system prompt for the agent."""
        if self.instructions is None:
            return None
        if isinstance(self.instructions, str):
            return self.instructions
        if callable(self.instructions):
            if inspect.iscoroutinefunction(self.instructions):
                return await cast(Awaitable[str], self.instructions(run_context, self))
            return cast(str, self.instructions(run_context, self))
        raise TypeError(f"Invalid instructions type: {type(self.instructions)}")
```

<br>

### creating a simple agent

<br>

```python
agent = Agent(
        name="Agent Dr. Love", 
        instructions="You are a cool special robot who loves")

setup_client()

result = Runner.run_sync(
            agent, 
            "Write a haiku about love in the cypherpunk world.")
```

<br>

### agent's gear

<br>

agents can be created with:

- a configurable instructions and model settings
- swords, for performing specific actions
- shields, for safety and validation of inputs and outputs
- orbs, to other agents or handlers
- charms, for monitoring and modifying an agent's behavior

<br>


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
        name_override=sword_name or _json.transform_string_function_style(agent.name),
        description_override=sword_description or "",
    )
    async def run_agent(context: RunContextWrapper, input: str) -> str:
        from .run import Runner

        output = await Runner.run(
            starting_agent=agent,
            input=input,
            context=context.context,
        )
        if custom_output_extractor:
            return await custom_output_extractor(output)

        return ItemHelpers.text_message_outputs(output.new_items)

    return run_agent
```
