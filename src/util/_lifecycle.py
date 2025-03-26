from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol

from ..agents.agent import Agent
from ._run_context import RunContextWrapper, TContext
from ._tool import Tool

########################################################
#             Class Run Hooks
########################################################

class HookProtocol(Protocol[TContext]):
    """Protocol defining the interface for lifecycle hooks."""
    
    async def on_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext]
    ) -> None: ...
    
    async def on_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any
    ) -> None: ...
    
    async def on_tool_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool
    ) -> None: ...
    
    async def on_tool_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str
    ) -> None: ...

class BaseHooks(ABC, Generic[TContext], HookProtocol[TContext]):
    """Base class for lifecycle hooks with common method signatures."""

    async def on_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext]
    ) -> None:
        """Called before agent invocation."""
        pass

    async def on_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any
    ) -> None:
        """Called when agent produces final output."""
        pass

    async def on_tool_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool
    ) -> None:
        """Called before tool invocation."""
        pass

    async def on_tool_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str
    ) -> None:
        """Called after tool invocation."""
        pass

class RunHooks(BaseHooks[TContext]):
    """Receives callbacks for agent run lifecycle events. Override methods as needed."""

    async def on_handoff(
        self,
        context: RunContextWrapper[TContext],
        from_agent: Agent[TContext],
        to_agent: Agent[TContext],
    ) -> None:
        """Called during agent handoff."""
        pass

########################################################
#             Class Agent Hooks
########################################################

class AgentHooks(BaseHooks[TContext]):
    """Receives callbacks for specific agent lifecycle events. Set on agent.hooks."""

    async def on_handoff(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], source: Agent[TContext]
    ) -> None:
        """Called when agent receives handoff. Source is the handing-off agent."""
        pass
