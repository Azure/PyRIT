# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest


def chunk_prompts(prompts: list[str] | list[NormalizerRequest], size: int):
    """
    Helper function utilized during prompt batching to chunk based off of size.

    Args:
        prompts (list[str] | list[NormalizerRequest]): List of prompts or Normalizer Requests to batch
        size (int): Batch size

    Returns:

    """
    for i in range(0, len(prompts), size):
        yield prompts[i : i + size]
