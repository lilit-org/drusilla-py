from __future__ import annotations

from typing import Any, Generic

from ..agents.agent import Agent
from ._run_context import RunContextWrapper, TContext
from ._tool import Tool

########################################################
#             Class Run Hooks
########################################################

class RunHooks(Generic[TContext]):
    """Receives callbacks for agent run lifecycle events. Override methods as needed."""

    async def on_agent_start(
        self, _context: RunContextWrapper[TContext], agent: Agent[TContext]
    ) -> None:
        """Called before agent invocation. Triggers on agent changes."""
        pass

    async def on_agent_end(
        self, _context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any
    ) -> None:
        """Called when agent produces final output."""
        pass

    async def on_handoff(
        self,
        _context: RunContextWrapper[TContext],
        from_agent: Agent[TContext],
        to_agent: Agent[TContext],
    ) -> None:
        """Called during agent handoff."""
        pass

    async def on_tool_start(
        self, _context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool
    ) -> None:
        """Called before tool invocation."""
        pass

    async def on_tool_end(
        self, _context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str
    ) -> None:
        """Called after tool invocation."""
        pass

########################################################
#             Class Agent Hooks
########################################################

class AgentHooks(Generic[TContext]):
    """Receives callbacks for specific agent lifecycle events. Set on agent.hooks."""

    async def on_start(
        self, _context: RunContextWrapper[TContext], agent: Agent[TContext]
    ) -> None:
        """Called before agent invocation. Triggers when agent becomes active."""
        pass

    async def on_end(
        self, _context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any
    ) -> None:
        """Called when agent produces final output."""
        pass

    async def on_handoff(
        self, _context: RunContextWrapper[TContext], agent: Agent[TContext], source: Agent[TContext]
    ) -> None:
        """Called when agent receives handoff. Source is the handing-off agent."""
        pass

    async def on_tool_start(
        self, _context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool
    ) -> None:
        """Called before tool invocation."""
        pass

    async def on_tool_end(
        self, _context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str
    ) -> None:
        """Called after tool invocation."""
        pass
