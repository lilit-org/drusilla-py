# charms

<br>

## tl; dr

<br>

charms are hooks that act as intelligent observers that can:

- intercept and modify agent behavior at key lifecycle points
- implement cross-cutting concerns like logging, monitoring, and validation
- enable sophisticated agent orchestration and coordination
- provide hooks for custom behavior injection before and after agent operations

<br>

---

## contents

<br>

- [overview of the charms module](#overview-of-the-charms-module)
    - [the `BaseCharms` dataclass](#the-basecharms-dataclass)
    - [the `RunCharms` dataclass](#the-runcharms-dataclass)
    - [the `AgentCharms` dataclass](#the-agentcharms-dataclass)
- [tips and best practices](#tips-and-best-practices)
  - [running tests](#running-tests)
- [available examples](#available-examples)

<br>

---

## overview of the charms module

<br>

### the `BaseCharms` dataclass

<br>

```python
class BaseCharms(ABC, Generic[TContext]):
    """Base class for lifecycle charms with common method signatures."""

    async def on_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext]) -> None:
        """Called before agent invocation."""

    async def on_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any
    ) -> None:
        """Called when agent produces final output."""

    async def on_sword_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], sword: Sword
    ) -> None:
        """Called before sword invocation."""

    async def on_sword_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        sword: Sword,
        result: Any,
    ) -> None:
        """Called after sword invocation."""
```

<br>

---

### the `RunCharms` dataclass

<br>

the only modification to the base class is `on_orbs()`, which in this case is called when the agent acts as an `Orb` object:

<br>

```python
class RunCharms(BaseCharms[TContext]):
    """Receives callbacks for agent run lifecycle events."""

    async def on_orbs(
        self,
        context: RunContextWrapper[TContext],
        from_agent: Agent[TContext],
        to_agent: Agent[TContext],
    ) -> None:
        """Called during agent orbs."""
```

<br>

---

### the `AgentCharms` dataclass

<br>

agent charms are mostly used to capture agents' lifecycles, for instance to fetch prior data or log events.

similarly, the only modification to the base class is also `on_orbs()`, called when the agent receives orbs:

<br>

```python
class AgentCharms(BaseCharms[TContext]):
    """Receives callbacks for specific agent lifecycle events."""

    async def on_orbs(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        source: Agent[TContext],
    ) -> None:
        """Called when agent receives orbs."""
```

<br>

----

## tips and best practices

<br>

### running tests

<br>

unit tests for the `Charms` module can be run with:

<br>

```shell
poetry run pytest tests/gear/test_charms.py -v
```

<br>

---

## available examples
