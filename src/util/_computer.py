import abc
from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

from ._constants import Button, Environment

########################################################
#               Type Variables                         #
########################################################

T = TypeVar("T", bound="BaseComputer")


########################################################
#               Base Computer Protocol                 #
########################################################


@runtime_checkable
class BaseComputer(Protocol):
    """Base interface for computer control operations."""

    @property
    @abc.abstractmethod
    def environment(self) -> Environment:
        """Get the current environment."""

    @property
    @abc.abstractmethod
    def dimensions(self) -> tuple[int, int]:
        """Get the screen dimensions."""

    @abc.abstractmethod
    def screenshot(self) -> str:
        """Take a screenshot and return it as a string."""

    @abc.abstractmethod
    def click(self, x: int, y: int, button: Button) -> None:
        """Click at the specified coordinates with the given button."""

    @abc.abstractmethod
    def double_click(self, x: int, y: int) -> None:
        """Double click at the specified coordinates."""

    @abc.abstractmethod
    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Scroll at the specified coordinates by the given amount."""

    @abc.abstractmethod
    def type(self, text: str) -> None:
        """Type the given text."""

    @abc.abstractmethod
    def wait(self) -> None:
        """Wait for any pending operations to complete."""

    @abc.abstractmethod
    def move(self, x: int, y: int) -> None:
        """Move the cursor to the specified coordinates."""

    @abc.abstractmethod
    def keypress(self, keys: Sequence[str]) -> None:
        """Press the specified keys."""

    @abc.abstractmethod
    def drag(self, path: Sequence[tuple[int, int]]) -> None:
        """Perform a drag operation along the specified path."""


########################################################
#               Synchronous Computer                   #
########################################################


class _BaseComputerMixin:
    """Shared validation methods for computer implementations."""

    def _validate_coordinates(self, x: int, y: int) -> None:
        width, height = self.dimensions
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Coordinates ({x}, {y}) outside screen bounds ({width}x{height})")

    def _validate_path(self, path: Sequence[tuple[int, int]]) -> None:
        if not path:
            raise ValueError("Empty path")
        for x, y in path:
            self._validate_coordinates(x, y)

    def _validate_keys(self, keys: Sequence[str]) -> None:
        if not keys:
            raise ValueError("Empty keys sequence")

    def _validate_text(self, text: str) -> None:
        if not text:
            raise ValueError("Empty text")


class Computer(_BaseComputerMixin, BaseComputer, abc.ABC):
    """Synchronous interface for controlling a computer or browser."""

    def screenshot(self) -> str:
        raise NotImplementedError("Screenshot not implemented")

    def click(self, x: int, y: int, button: Button) -> None:
        self._validate_coordinates(x, y)
        raise NotImplementedError("Click not implemented")

    def double_click(self, x: int, y: int) -> None:
        self._validate_coordinates(x, y)
        raise NotImplementedError("Double click not implemented")

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self._validate_coordinates(x, y)
        raise NotImplementedError("Scroll not implemented")

    def type(self, text: str) -> None:
        self._validate_text(text)
        raise NotImplementedError("Type not implemented")

    def wait(self) -> None:
        raise NotImplementedError("Wait not implemented")

    def move(self, x: int, y: int) -> None:
        self._validate_coordinates(x, y)
        raise NotImplementedError("Move not implemented")

    def keypress(self, keys: Sequence[str]) -> None:
        self._validate_keys(keys)
        raise NotImplementedError("Keypress not implemented")

    def drag(self, path: Sequence[tuple[int, int]]) -> None:
        self._validate_path(path)
        raise NotImplementedError("Drag not implemented")


########################################################
#               Asynchronous Computer                   #
########################################################


class AsyncComputer(_BaseComputerMixin, BaseComputer, abc.ABC):
    """Asynchronous interface for controlling a computer or browser."""

    @abc.abstractmethod
    async def screenshot(self) -> str:
        raise NotImplementedError("Screenshot not implemented")

    @abc.abstractmethod
    async def click(self, x: int, y: int, button: Button) -> None:
        self._validate_coordinates(x, y)
        raise NotImplementedError("Click not implemented")

    @abc.abstractmethod
    async def double_click(self, x: int, y: int) -> None:
        self._validate_coordinates(x, y)
        raise NotImplementedError("Double click not implemented")

    @abc.abstractmethod
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self._validate_coordinates(x, y)
        raise NotImplementedError("Scroll not implemented")

    @abc.abstractmethod
    async def type(self, text: str) -> None:
        self._validate_text(text)
        raise NotImplementedError("Type not implemented")

    @abc.abstractmethod
    async def wait(self) -> None:
        raise NotImplementedError("Wait not implemented")

    @abc.abstractmethod
    async def move(self, x: int, y: int) -> None:
        self._validate_coordinates(x, y)
        raise NotImplementedError("Move not implemented")

    @abc.abstractmethod
    async def keypress(self, keys: Sequence[str]) -> None:
        self._validate_keys(keys)
        raise NotImplementedError("Keypress not implemented")

    @abc.abstractmethod
    async def drag(self, path: Sequence[tuple[int, int]]) -> None:
        self._validate_path(path)
        raise NotImplementedError("Drag not implemented")
