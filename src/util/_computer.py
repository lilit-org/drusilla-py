import abc
from typing import Literal

########################################################
#              Constants
########################################################

Environment = Literal["mac", "windows", "ubuntu", "browser"]
Button = Literal["left", "right", "wheel", "back", "forward"]


########################################################
#              Computer Interface
########################################################
class Computer(abc.ABC):
    """Sync interface for controlling a computer or browser."""

    @property
    @abc.abstractmethod
    def environment(self) -> Environment:
        pass

    @property
    @abc.abstractmethod
    def dimensions(self) -> tuple[int, int]:
        pass

    @abc.abstractmethod
    def screenshot(self) -> str:
        pass

    @abc.abstractmethod
    def click(self, x: int, y: int, button: Button) -> None:
        pass

    @abc.abstractmethod
    def double_click(self, x: int, y: int) -> None:
        pass

    @abc.abstractmethod
    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        pass

    @abc.abstractmethod
    def type(self, text: str) -> None:
        pass

    @abc.abstractmethod
    def wait(self) -> None:
        pass

    @abc.abstractmethod
    def move(self, x: int, y: int) -> None:
        pass

    @abc.abstractmethod
    def keypress(self, keys: list[str]) -> None:
        pass

    @abc.abstractmethod
    def drag(self, path: list[tuple[int, int]]) -> None:
        pass


########################################################
#              Async Computer Interface
########################################################
class AsyncComputer(abc.ABC):
    """Async interface for controlling a computer or browser."""

    @property
    @abc.abstractmethod
    def environment(self) -> Environment:
        pass

    @property
    @abc.abstractmethod
    def dimensions(self) -> tuple[int, int]:
        pass

    @abc.abstractmethod
    async def screenshot(self) -> str:
        pass

    @abc.abstractmethod
    async def click(self, x: int, y: int, button: Button) -> None:
        pass

    @abc.abstractmethod
    async def double_click(self, x: int, y: int) -> None:
        pass

    @abc.abstractmethod
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        pass

    @abc.abstractmethod
    async def type(self, text: str) -> None:
        pass

    @abc.abstractmethod
    async def wait(self) -> None:
        pass

    @abc.abstractmethod
    async def move(self, x: int, y: int) -> None:
        pass

    @abc.abstractmethod
    async def keypress(self, keys: list[str]) -> None:
        pass

    @abc.abstractmethod
    async def drag(self, path: list[tuple[int, int]]) -> None:
        pass
