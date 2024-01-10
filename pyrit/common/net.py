# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import aiohttp


class HttpClientSession:
    """AioHttp client session which encapsulates TCP connection pool and alive always
    while service is running to benefit connection pooling

    Returns:
        [aiohttp.ClientSession]: AioHttp client session object hold TCP connections
        can be used across all Http Get and Post requests
    """

    # create aiohttp client session once and re-use for all the HTTP requests
    # that defaults to 100 TCP connections per client session
    client_session: aiohttp.ClientSession = None

    @staticmethod
    def get_client_session() -> aiohttp.ClientSession:
        if not HttpClientSession.client_session:
            HttpClientSession.client_session = aiohttp.ClientSession()
        return HttpClientSession.client_session
