"""
This module provides a default async HTTP client for making requests.
It extends the httpx.AsyncClient with additional features:

1. Automatic retry mechanism for failed requests with exponential backoff
2. Configurable timeouts for different phases of the request
3. Connection pooling and limits management
4. HTTP/2 support
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from httpx import Limits, Timeout

from ..util.constants import config, err
from ..util.exceptions import ConnectionError

########################################################
#              Main class: Async HTTP Client
########################################################


class DefaultAsyncHttpxClient(httpx.AsyncClient):

    def __init__(
        self,
        *,
        timeout: float | Timeout | None = config.HTTP_TIMEOUT_TOTAL,
        connect_timeout: float = config.HTTP_TIMEOUT_CONNECT,
        read_timeout: float = config.HTTP_TIMEOUT_READ,
        max_keepalive_connections: int = config.HTTP_MAX_KEEPALIVE_CONNECTIONS,
        max_connections: int = config.HTTP_MAX_CONNECTIONS,
        max_retries: int = 3,
        verify: bool | str = True,
        http2: bool = False,
        **kwargs: Any,
    ) -> None:

        timeout = (
            Timeout(timeout=timeout, connect=connect_timeout, read=read_timeout)
            if isinstance(timeout, int | float)
            else timeout
        )

        self.limits = Limits(
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
        )

        self.max_retries = max_retries
        self.verify = verify
        self.http2 = http2

        super().__init__(
            timeout=timeout,
            verify=verify,
            http2=http2,
            limits=self.limits,
            **kwargs,
        )

    async def request(self, *args, **kwargs) -> httpx.Response:
        for attempt in range(self.max_retries):
            try:
                return await super().request(*args, **kwargs)
            except (httpx.ConnectError, httpx.ReadError) as e:
                if attempt == self.max_retries - 1:
                    raise ConnectionError(
                        err.AGENT_EXEC_ERROR.format(
                            error=f"Connection failed after {self.max_retries} attempts: {str(e)}"
                        )
                    ) from e

                # Exponential backoff with jitter
                backoff = min(2**attempt + (attempt * 0.1), 30)  # Cap at 30 seconds
                await asyncio.sleep(backoff)

    async def get(self, *args: Any, **kwargs: Any) -> httpx.Response:
        """Make a GET request with automatic retries."""
        return await self.request("GET", *args, **kwargs)

    async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
        """Make a POST request with automatic retries."""
        return await self.request("POST", *args, **kwargs)

    async def put(self, *args: Any, **kwargs: Any) -> httpx.Response:
        """Make a PUT request with automatic retries."""
        return await self.request("PUT", *args, **kwargs)

    async def delete(self, *args: Any, **kwargs: Any) -> httpx.Response:
        """Make a DELETE request with automatic retries."""
        return await self.request("DELETE", *args, **kwargs)
