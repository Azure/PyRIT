# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Callable


def set_max_requests_per_minute(func: Callable) -> Callable:
    """
    A decorator to introduce a delay in between requests to ensure rate limit of the target is respected.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function with a sleep introduced.
    """

    async def set_max_rpm(*args, **kwargs):
        rpm = args[0]._requests_per_minute  # args[0] will be 'self' from the calling target
        if rpm:
            await asyncio.sleep(60 / rpm)

        return await func(*args, **kwargs)

    return set_max_rpm
