# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import random
from typing import List, Union


def combine_dict(existing_dict: dict = None, new_dict: dict = None) -> dict:
    """
    Combines two dictionaries containing string keys and values into one.

    Args:
        existing_dict: Dictionary with existing values
        new_dict: Dictionary with new values to be added to the existing dictionary.
            Note if there's a key clash, the value in new_dict will be used.

    Returns:
        dict: combined dictionary
    """
    result = {**(existing_dict or {})}
    result.update(new_dict or {})
    return result


def combine_list(list1: Union[str, List[str]], list2: Union[str, List[str]]) -> list:
    """
    Combines two lists containing string keys, keeping only unique values.

    Args:
        existing_dict: Dictionary with existing values
        new_dict: Dictionary with new values to be added to the existing dictionary.
            Note if there's a key clash, the value in new_dict will be used.

    Returns:
        list: combined dictionary
    """
    if isinstance(list1, str):
        list1 = [list1]
    if isinstance(list2, str):
        list2 = [list2]

    # Merge and keep only unique values
    combined = list(set(list1 + list2))
    return combined


def get_random_indices(*, start: int, size: int, proportion: float) -> List[int]:
    """
    Generate a list of random indices based on the specified proportion of a given size.
    The indices are selected from the range [start, start + size).

    Args:
        start (int): Starting index (inclusive). It's the first index that could possibly be selected.
        size (int): Size of the collection to select from. This is the total number of indices available.
            For example, if `start` is 0 and `size` is 10, the available indices are [0, 1, 2, ..., 9].
        proportion (float): The proportion of indices to select from the total size. Must be between 0 and 1.
            For example, if `proportion` is 0.5 and `size` is 10, 5 randomly selected indices will be returned.

    Returns:
        List[int]: A list of randomly selected indices based on the specified proportion.
    """
    if start < 0:
        raise ValueError("Start index must be non-negative")
    if size <= 0:
        raise ValueError("Size must be greater than 0")
    if proportion < 0 or proportion > 1:
        raise ValueError("Proportion must be between 0 and 1")

    if proportion == 0:
        return []
    if proportion == 1:
        return list(range(start, start + size))

    n = max(math.ceil(size * proportion), 1)  # the number of indices to select
    return random.sample(range(start, start + size), n)
