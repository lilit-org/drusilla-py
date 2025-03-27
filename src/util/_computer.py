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


class Computer(BaseComputer, abc.ABC):
    """Synchronous interface for controlling a computer or browser."""

    def _validate_coordinates(self, x: int, y: int) -> None:
        """Validate that coordinates are within screen bounds."""
        width, height = self.dimensions
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Coordinates ({x}, {y}) are outside screen bounds ({width}x{height})"
            )

    def _validate_path(self, path: Sequence[tuple[int, int]]) -> None:
        """Validate that all points in a path are within screen bounds."""
        if not path:
            raise ValueError("Path cannot be empty")
        for x, y in path:
            self._validate_coordinates(x, y)

    def _validate_keys(self, keys: Sequence[str]) -> None:
        """Validate that the keys sequence is not empty."""
        if not keys:
            raise ValueError("Keys sequence cannot be empty")

    def screenshot(self) -> str:
        """Take a screenshot and return it as a string."""
        raise NotImplementedError(
            "Screenshot method must be implemented by concrete class"
        )

    def click(self, x: int, y: int, button: Button) -> None:
        """Click at the specified coordinates with the given button."""
        self._validate_coordinates(x, y)
        raise NotImplementedError("Click method must be implemented by concrete class")

    def double_click(self, x: int, y: int) -> None:
        """Double click at the specified coordinates."""
        self._validate_coordinates(x, y)
        raise NotImplementedError(
            "Double click method must be implemented by concrete class"
        )

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Scroll at the specified coordinates by the given amount."""
        self._validate_coordinates(x, y)
        raise NotImplementedError("Scroll method must be implemented by concrete class")

    def type(self, text: str) -> None:
        """Type the given text."""
        if not text:
            raise ValueError("Text cannot be empty")
        raise NotImplementedError("Type method must be implemented by concrete class")

    def wait(self) -> None:
        """Wait for any pending operations to complete."""
        raise NotImplementedError("Wait method must be implemented by concrete class")

    def move(self, x: int, y: int) -> None:
        """Move the cursor to the specified coordinates."""
        self._validate_coordinates(x, y)
        raise NotImplementedError("Move method must be implemented by concrete class")

    def keypress(self, keys: Sequence[str]) -> None:
        """Press the specified keys."""
        self._validate_keys(keys)
        raise NotImplementedError(
            "Keypress method must be implemented by concrete class"
        )

    def drag(self, path: Sequence[tuple[int, int]]) -> None:
        """Perform a drag operation along the specified path."""
        self._validate_path(path)
        raise NotImplementedError("Drag method must be implemented by concrete class")


########################################################
#               Asynchronous Computer                   #
########################################################


class AsyncComputer(BaseComputer, abc.ABC):
    """Asynchronous interface for controlling a computer or browser."""

    def _validate_coordinates(self, x: int, y: int) -> None:
        """Validate that coordinates are within screen bounds."""
        width, height = self.dimensions
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Coordinates ({x}, {y}) are outside screen bounds ({width}x{height})"
            )

    def _validate_path(self, path: Sequence[tuple[int, int]]) -> None:
        """Validate that all points in a path are within screen bounds."""
        if not path:
            raise ValueError("Path cannot be empty")
        for x, y in path:
            self._validate_coordinates(x, y)

    def _validate_keys(self, keys: Sequence[str]) -> None:
        """Validate that the keys sequence is not empty."""
        if not keys:
            raise ValueError("Keys sequence cannot be empty")

    @abc.abstractmethod
    async def screenshot(self) -> str:
        """Take a screenshot and return it as a string."""
        raise NotImplementedError(
            "Screenshot method must be implemented by concrete class"
        )

    @abc.abstractmethod
    async def click(self, x: int, y: int, button: Button) -> None:
        """Click at the specified coordinates with the given button."""
        self._validate_coordinates(x, y)
        raise NotImplementedError("Click method must be implemented by concrete class")

    @abc.abstractmethod
    async def double_click(self, x: int, y: int) -> None:
        """Double click at the specified coordinates."""
        self._validate_coordinates(x, y)
        raise NotImplementedError(
            "Double click method must be implemented by concrete class"
        )

    @abc.abstractmethod
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Scroll at the specified coordinates by the given amount."""
        self._validate_coordinates(x, y)
        raise NotImplementedError("Scroll method must be implemented by concrete class")

    @abc.abstractmethod
    async def type(self, text: str) -> None:
        """Type the given text."""
        if not text:
            raise ValueError("Text cannot be empty")
        raise NotImplementedError("Type method must be implemented by concrete class")

    @abc.abstractmethod
    async def wait(self) -> None:
        """Wait for any pending operations to complete."""
        raise NotImplementedError("Wait method must be implemented by concrete class")

    @abc.abstractmethod
    async def move(self, x: int, y: int) -> None:
        """Move the cursor to the specified coordinates."""
        self._validate_coordinates(x, y)
        raise NotImplementedError("Move method must be implemented by concrete class")

    @abc.abstractmethod
    async def keypress(self, keys: Sequence[str]) -> None:
        """Press the specified keys."""
        self._validate_keys(keys)
        raise NotImplementedError(
            "Keypress method must be implemented by concrete class"
        )

    @abc.abstractmethod
    async def drag(self, path: Sequence[tuple[int, int]]) -> None:
        """Perform a drag operation along the specified path."""
        self._validate_path(path)
        raise NotImplementedError("Drag method must be implemented by concrete class")
