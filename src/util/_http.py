from __future__ import annotations

import asyncio
from typing import Any

import httpx
from httpx import Limits, Timeout


class DefaultAsyncHttpxClient(httpx.AsyncClient):
    """A default async HTTP client for making requests."""

    def __init__(
        self,
        *,
        timeout: float | Timeout | None = 120.0,
        connect_timeout: float = 30.0,
        read_timeout: float = 90.0,
        max_keepalive_connections: int = 5,
        max_connections: int = 10,
        max_retries: int = 3,
        verify: bool | str = True,
        http2: bool = True,
        **kwargs: Any,
    ) -> None:
        if isinstance(timeout, (int, float)):
            timeout = Timeout(
                timeout=timeout,
                connect=connect_timeout,
                read=read_timeout,
            )
        limits = Limits(
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
        )

        super().__init__(
            timeout=timeout,
            verify=verify,
            http2=http2,
            limits=limits,
            **kwargs,
        )

        self.max_retries = max_retries

    async def request(self, *args: Any, **kwargs: Any) -> httpx.Response:
        """Send a request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return await super().request(*args, **kwargs)
            except (httpx.ConnectError, httpx.ReadError):
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
