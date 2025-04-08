"""
Charms Module - Agent Lifecycle Management

This module implements an event-driven system for managing agent
lifecycles through Charms. Charms act as intelligent observers that can:

- Intercept and modify agent behavior at key lifecycle points
- Implement cross-cutting concerns like logging, monitoring, and validation
- Enable sophisticated agent orchestration and coordination
- Provide hooks for custom behavior injection before and after agent operations
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Generic

from ..agents.agent_v1 import AgentV1 as Agent
from ..util.types import RunContextWrapper, TContext
from .sword import Sword

########################################################
#             Main Classes for Charms
########################################################


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


class RunCharms(BaseCharms[TContext]):
    """Receives callbacks for agent run lifecycle events."""

    async def on_orbs(
        self,
        context: RunContextWrapper[TContext],
        from_agent: Agent[TContext],
        to_agent: Agent[TContext],
    ) -> None:
        """Called during agent orbs."""


class AgentCharms(BaseCharms[TContext]):
    """Receives callbacks for specific agent lifecycle events."""

    async def on_orbs(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        source: Agent[TContext],
    ) -> None:
        """Called when agent receives orbs."""
