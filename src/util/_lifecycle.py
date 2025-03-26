from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic

from ..agents.agent import Agent
from ._run_context import RunContextWrapper, TContext
from ._tool import Tool

########################################################
#             Class Run Hooks
########################################################

class BaseHooks(ABC, Generic[TContext]):
    """Base class for lifecycle hooks with common method signatures."""

    @abstractmethod
    async def on_start(
        self, _context: RunContextWrapper[TContext], _agent: Agent[TContext]
    ) -> None:
        """Called before agent invocation."""
        pass

    @abstractmethod
    async def on_end(
        self, _context: RunContextWrapper[TContext], _agent: Agent[TContext], _output: Any
    ) -> None:
        """Called when agent produces final output."""
        pass

    @abstractmethod
    async def on_tool_start(
        self, _context: RunContextWrapper[TContext], _agent: Agent[TContext], _tool: Tool
    ) -> None:
        """Called before tool invocation."""
        pass

    @abstractmethod
    async def on_tool_end(
        self, _context: RunContextWrapper[TContext], _agent: Agent[TContext], _tool: Tool, _result: str
    ) -> None:
        """Called after tool invocation."""
        pass

class RunHooks(BaseHooks[TContext]):
    """Receives callbacks for agent run lifecycle events. Override methods as needed."""

    async def on_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext]
    ) -> None:
        """Called before agent invocation. Triggers on agent changes."""
        pass

    async def on_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any
    ) -> None:
        """Called when agent produces final output."""
        pass

    async def on_handoff(
        self,
        context: RunContextWrapper[TContext],
        from_agent: Agent[TContext],
        to_agent: Agent[TContext],
    ) -> None:
        """Called during agent handoff."""
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

########################################################
#             Class Agent Hooks
########################################################

class AgentHooks(BaseHooks[TContext]):
    """Receives callbacks for specific agent lifecycle events. Set on agent.hooks."""

    async def on_start(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext]
    ) -> None:
        """Called before agent invocation. Triggers when agent becomes active."""
        pass

    async def on_end(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any
    ) -> None:
        """Called when agent produces final output."""
        pass

    async def on_handoff(
        self, context: RunContextWrapper[TContext], agent: Agent[TContext], source: Agent[TContext]
    ) -> None:
        """Called when agent receives handoff. Source is the handing-off agent."""
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
