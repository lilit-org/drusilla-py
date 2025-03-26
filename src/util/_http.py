from __future__ import annotations

from typing import Any

import httpx


class DefaultAsyncHttpxClient(httpx.AsyncClient):
    """A default async HTTP client for making requests."""

    def __init__(
        self,
        *,
        timeout: float | httpx.Timeout | None = 60.0,
        verify: bool | str = True,
        http2: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            timeout=timeout,
            verify=verify,
            http2=http2,
            **kwargs,
        )
