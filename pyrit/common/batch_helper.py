# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Callable

from pyrit.prompt_target import PromptTarget


def _chunked_prompts(prompts, batch_size: int):
    """
    Helper function utilized during prompt batching to chunk based off of size.

    Args:
        prompts (list[str, NormalizerRequest, PromptRequestPiece]): List of prompts to batch
        batch_size (int): Batch size

    """
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]


def _validate_rate_limit_parameters(prompt_target: PromptTarget, batch_size: int):
    """
    Helper function to validate the constraints between Rate Limit (Requests Per Minute)
        and batch size.

    Args:
        prompt_target (PromptTarget): Target to validate
        batch_size (int): Batch size

    Raises:
        ValueError: When rate limit RPM is specified for the target and batching is not adjusted to 1.
    """

    exc_message = "Batch size must be configured to 1 for the target requests per minute value to be respected."
    if prompt_target and prompt_target._max_requests_per_minute and batch_size != 1:
        raise ValueError(exc_message)


async def batch_task_async(
    *, prompt_target: PromptTarget, batch_size: int, items_to_batch, task: Callable, task_argument: str, **task_kwargs
):
    """
    Performs provided task in batches and validates parameters using helpers.

    Args:
        prompt_target(PromptTarget): Target to validate
        batch_size (int): Batch size
        items_to_batch (list): List of items to batch
        task (Callable): Task to perform in batches
        task_argument (str): Name of argument to assign prompt to
        **task_kwargs: Any other keyword arguments that task needs

    Returns:
        responses(list): List of results from the batched function
    """

    responses = []

    _validate_rate_limit_parameters(prompt_target=prompt_target, batch_size=batch_size)

    for prompts_batch in _chunked_prompts(items_to_batch, batch_size):
        tasks = []
        for prompt in prompts_batch:
            task_kwargs[task_argument] = prompt
            tasks.append(task(**task_kwargs))

        batch_results = await asyncio.gather(*tasks)
        responses.extend(batch_results)

    return responses
