# primitives

<br>

the drusilla project has a minimal set of core primitives that allows the design of multi-agent systems:

<br>

- [agents](agents.md): LLM (ro)bots that can be equipped with orbs and shields
- [runners](runners.md): execution environments that run agents and manage their lifecycle
- [models](models.md): settings and classes to handle the agent's LLM model
- [agents' gear](gear/)
    - [sword](gear/sword.md): part of the agents' gear, used to let agents take actions
    - [orbs](gear/orbs.md): part of the agents' gear, used to delegate tasks to other agents
    - [shield](gear/shield.md): part of the agents' gear, used to validate and protect the inputs from agents
    - [charms](gear/charms.md): part of the agents' gear, used to receive callbacks on lifecycle events
- [mcp](mcp.md): a paradigm to provide external capabilities, tools, and context for an LLM
