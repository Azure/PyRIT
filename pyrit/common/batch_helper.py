# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def chunk_prompts(prompts: list[str], size: int):
    """
    Helper function utilized during prompt batching to chunk based off of size.

    Args:
        prompts (list[str]): List of prompts to batch,
        size (int): Size of batch.

    Returns:

    """
    for i in range(0, len(prompts), size):
        yield prompts[i : i + size]