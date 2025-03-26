import abc
from typing import Protocol, TypeVar

from ._constants import Button, Environment

########################################################
#               Type Variables                         #
########################################################

T = TypeVar('T', bound='BaseComputer')


########################################################
#               Base Computer Protocol                 #
########################################################

class BaseComputer(Protocol):
    """Base interface for computer control operations."""

    @property
    @abc.abstractmethod
    def environment(self) -> Environment:
        """Get the current environment."""
        pass

    @property
    @abc.abstractmethod
    def dimensions(self) -> tuple[int, int]:
        """Get the screen dimensions."""
        pass

    @abc.abstractmethod
    def screenshot(self) -> str:
        """Take a screenshot and return it as a string."""
        pass

    @abc.abstractmethod
    def click(self, x: int, y: int, button: Button) -> None:
        """Click at the specified coordinates with the given button."""
        pass

    @abc.abstractmethod
    def double_click(self, x: int, y: int) -> None:
        """Double click at the specified coordinates."""
        pass

    @abc.abstractmethod
    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Scroll at the specified coordinates by the given amount."""
        pass

    @abc.abstractmethod
    def type(self, text: str) -> None:
        """Type the given text."""
        pass

    @abc.abstractmethod
    def wait(self) -> None:
        """Wait for any pending operations to complete."""
        pass

    @abc.abstractmethod
    def move(self, x: int, y: int) -> None:
        """Move the cursor to the specified coordinates."""
        pass

    @abc.abstractmethod
    def keypress(self, keys: list[str]) -> None:
        """Press the specified keys."""
        pass

    @abc.abstractmethod
    def drag(self, path: list[tuple[int, int]]) -> None:
        """Perform a drag operation along the specified path."""
        pass


########################################################
#               Synchronous Computer                   #
########################################################

class Computer(BaseComputer, abc.ABC):
    """Synchronous interface for controlling a computer or browser."""


########################################################
#               Asynchronous Computer                   #
########################################################

class AsyncComputer(BaseComputer, abc.ABC):
    """Asynchronous interface for controlling a computer or browser."""

    @abc.abstractmethod
    async def screenshot(self) -> str:
        """Take a screenshot and return it as a string."""
        pass

    @abc.abstractmethod
    async def click(self, x: int, y: int, button: Button) -> None:
        """Click at the specified coordinates with the given button."""
        pass

    @abc.abstractmethod
    async def double_click(self, x: int, y: int) -> None:
        """Double click at the specified coordinates."""
        pass

    @abc.abstractmethod
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Scroll at the specified coordinates by the given amount."""
        pass

    @abc.abstractmethod
    async def type(self, text: str) -> None:
        """Type the given text."""
        pass

    @abc.abstractmethod
    async def wait(self) -> None:
        """Wait for any pending operations to complete."""
        pass

    @abc.abstractmethod
    async def move(self, x: int, y: int) -> None:
        """Move the cursor to the specified coordinates."""
        pass

    @abc.abstractmethod
    async def keypress(self, keys: list[str]) -> None:
        """Press the specified keys."""
        pass

    @abc.abstractmethod
    async def drag(self, path: list[tuple[int, int]]) -> None:
        """Perform a drag operation along the specified path."""
        pass
