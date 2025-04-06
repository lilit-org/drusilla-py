from __future__ import annotations

from abc import ABC
from typing import Any, Generic, Protocol

from ..agents.agent import Agent
from ..gear.swords import Sword
from ._run_context import RunContextWrapper, TContext

########################################################
#             Main Classes for Lifecycle Hooks
########################################################


class HookProtocol(Protocol[TContext]):
    """Protocol defining the interface for lifecycle hooks."""

    async def on_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext]
    ) -> None: ...

    async def on_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any
    ) -> None: ...

    async def on_sword_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], sword: Sword
    ) -> None: ...

    async def on_sword_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        sword: Sword,
        result: Any,
    ) -> None: ...


class BaseHooks(ABC, Generic[TContext], HookProtocol[TContext]):
    """Base class for lifecycle hooks with common method signatures."""

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


class RunHooks(BaseHooks[TContext]):
    """Receives callbacks for agent run lifecycle events."""

    async def on_orbs(
        self,
        context: RunContextWrapper[TContext],
        from_agent: Agent[TContext],
        to_agent: Agent[TContext],
    ) -> None:
        """Called during agent orbs."""


class AgentHooks(BaseHooks[TContext]):
    """Receives callbacks for specific agent lifecycle events."""

    async def on_orbs(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        source: Agent[TContext],
    ) -> None:
        """Called when agent receives orbs."""
