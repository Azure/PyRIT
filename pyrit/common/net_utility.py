# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed


def get_httpx_client(use_async: bool = False, debug: bool = False) -> httpx.Client:
    """Get the httpx client for making requests."""

    client_class = httpx.AsyncClient if use_async else httpx.Client
    proxies = "http://localhost:8080" if debug else None
    verify_certs = not debug

    # fun notes; httpx default is 5 seconds, httpclient is 100, urllib in indefinite
    return client_class(proxies=proxies, verify=verify_certs, timeout=60.0)


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
def make_request_and_raise_if_error(
    endpoint_uri: str,
    method: str,
    request_body: dict[str, str] = None,
    headers: dict[str, str] = None,
    debug: bool = False,
) -> httpx.Response:
    """Make a request and raise an exception if it fails."""
    headers = headers or {}
    request_body = request_body or {}

    with get_httpx_client(debug=debug) as client:
        if request_body:
            headers["Content-Type"] = "application/json"
            response = client.request(method=method, url=endpoint_uri, json=request_body, headers=headers)
        else:
            response = client.request(method=method, url=endpoint_uri, headers=headers)

        response.raise_for_status()  # This will automatically raise an exception for 4xx and 5xx responses

    return response
