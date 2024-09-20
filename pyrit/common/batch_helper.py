# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Any, Callable, Sequence

from pyrit.prompt_target import PromptTarget


def _get_chunks(*args, batch_size: int):
    """
    Helper function utilized during prompt batching to chunk based off of size.

    Args:
        *args: Arguments to chunk; each argument should be a list
        batch_size (int): Batch size

    """
    if len(args) == 0:
        raise ValueError("No arguments provided to chunk.")
    for arg in args[1:]:
        if len(arg) != len(args[0]):
            raise ValueError("All arguments must have the same length.")
    for i in range(0, len(args[0]), batch_size):
        yield [arg[i : i + batch_size] for arg in args]


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
    *,
    prompt_target: PromptTarget,
    batch_size: int,
    items_to_batch: Sequence[Sequence[Any]],
    task_func: Callable,
    task_arguments: list[str],
    **task_kwargs,
):
    """
    Performs provided task in batches and validates parameters using helpers.

    Args:
        prompt_target(PromptTarget): Target to validate
        batch_size (int): Batch size
        items_to_batch (list[list[Any]]): Lists of items to batch
        task_func (Callable): Task to perform in batches
        task_arguments (list[str]): Name of arguments to assign lists of items to
        **task_kwargs: Any other keyword arguments that task needs

    Returns:
        responses(list): List of results from the batched function
    """

    responses = []

    _validate_rate_limit_parameters(prompt_target=prompt_target, batch_size=batch_size)

    if len(items_to_batch) == 0 or len(items_to_batch[0]) == 0:
        raise ValueError("No items to batch.")

    if len(items_to_batch) != len(task_arguments):
        raise ValueError("Number of lists of items to batch must match number of task arguments.")

    for task_args in _get_chunks(*items_to_batch, batch_size=batch_size):
        tasks = []
        for batch_index in range(len(task_args[0])):
            for arg_index, task_argument in enumerate(task_arguments):
                task_kwargs[task_argument] = task_args[arg_index][batch_index]
            tasks.append(task_func(**task_kwargs))

        batch_results = await asyncio.gather(*tasks)
        responses.extend(batch_results)

    return responses
