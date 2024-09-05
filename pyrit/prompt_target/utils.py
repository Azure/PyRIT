# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Callable


def limit_requests_per_minute(func: Callable) -> Callable:
    """
    A decorator to enforce rate limit of the target through setting requests per minute.
    This should be applied to all send_prompt_async() functions on PromptTarget and PromptChatTarget.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function with a sleep introduced.
    """

    async def set_max_rpm(*args, **kwargs):
        self = args[0]
        rpm = getattr(self, "_max_requests_per_minute", None)
        if rpm and rpm > 0:
            await asyncio.sleep(60 / rpm)

        return await func(*args, **kwargs)

    return set_max_rpm
