# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Literal, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed


def get_httpx_client(use_async: bool = False, debug: bool = False, **httpx_client_kwargs: Optional[Any]):
    """Get the httpx client for making requests."""

    client_class = httpx.AsyncClient if use_async else httpx.Client
    proxy = "http://localhost:8080" if debug else None

    proxy = httpx_client_kwargs.pop("proxy", proxy)
    verify_certs = httpx_client_kwargs.pop("verify", not debug)
    # fun notes; httpx default is 5 seconds, httpclient is 100, urllib in indefinite
    timeout = httpx_client_kwargs.pop("timeout", 60.0)

    return client_class(proxy=proxy, verify=verify_certs, timeout=timeout, **httpx_client_kwargs)


PostType = Literal["json", "data"]


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
async def make_request_and_raise_if_error_async(
    endpoint_uri: str,
    method: str,
    params: dict[str, str] = None,
    request_body: dict[str, object] = None,
    headers: dict[str, str] = None,
    post_type: PostType = "json",
    debug: bool = False,
    **httpx_client_kwargs: Optional[Any],
) -> httpx.Response:
    """Make a request and raise an exception if it fails."""
    headers = headers or {}
    request_body = request_body or {}

    params = params or {}

    async with get_httpx_client(debug=debug, use_async=True, **httpx_client_kwargs) as async_client:
        response = await async_client.request(
            method=method,
            params=params,
            url=endpoint_uri,
            json=request_body if request_body and post_type == "json" else None,
            data=request_body if request_body and post_type != "json" else None,
            headers=headers,
        )

        response.raise_for_status()  # This will automatically raise an exception for 4xx and 5xx responses

    return response
