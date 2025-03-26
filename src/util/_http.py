from __future__ import annotations

from typing import Any

import httpx


class DefaultAsyncHttpxClient(httpx.AsyncClient):
    """A default async HTTP client for making requests."""

    def __init__(
        self,
        *,
        timeout: float | httpx.Timeout | None = 120.0,
        verify: bool | str = True,
        http2: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            timeout=httpx.Timeout(timeout, connect=30.0, read=90.0),
            verify=verify,
            http2=http2,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            **kwargs,
        )
