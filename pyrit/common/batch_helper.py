# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def chunk_prompts(prompts, size: int):
    """
    Helper function utilized during prompt batching to chunk based off of size.

    Args:
        prompts (list[str, NormalizerRequest, PromptRequestPiece]): List of prompts to batch
        size (int): Batch size

    Returns:

    """
    for i in range(0, len(prompts), size):
        yield prompts[i : i + size]
