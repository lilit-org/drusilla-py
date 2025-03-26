from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypeVar

from ._usage import Usage

########################################################
#              Data class RunContextWrapper            #
########################################################

TContext = TypeVar("TContext", default=Any)

@dataclass
class RunContextWrapper(Generic[TContext]):
    """Wrapper for context objects passed to Runner.run().

    Contexts are used to pass dependencies and data to custom code (tools, callbacks, hooks).
    They are not passed to the LLM.
    """

    context: TContext
    """Context object passed to Runner.run()"""

    usage: Usage = field(default_factory=Usage)
    """Usage stats for the agent run. May be stale during streaming until final chunk."""
