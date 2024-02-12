# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from urllib3.util.retry import Retry
from urllib3 import BaseHTTPResponse, PoolManager, ProxyManager

def get_pool_manager(retries: int = 3, use_proxy: bool = False) -> PoolManager:
    """Get the pool manager for the requests"""
    retry_strategy = Retry(
        total=retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "POST"],
        backoff_factor=1
    )

    if use_proxy:
        # Use ProxyManager if the use_proxy flag is True
        return ProxyManager(retries=retry_strategy, proxy_url="http://localhost:8080")
    else:
        # Use PoolManager if no proxy is needed
        return PoolManager(retries=retry_strategy)


# TODO make proxy false
def make_request_and_raise_if_error(endpoint_uri: str,
                           method: str,
                           json_body: dict[str, str],
                           headers: dict[str, str],
                           retries: int = 3,
                           use_proxy: bool = True) -> BaseHTTPResponse:
    """Make a request and raise an exception if it fails"""
    http = get_pool_manager(retries=retries, use_proxy=use_proxy)

    if json_body:
        json_body = json.dumps(json_body).encode('utf-8') if json_body else None
        headers["Content-Type"] = "application/json"

    response = http.request(method=method, url=endpoint_uri, body=json_body, headers=headers)

    if response.status >= 400:
        error_body = response.data.decode('utf-8')
        raise RuntimeError(f"HTTP error: {response.reason}\n{error_body}.")

    return response