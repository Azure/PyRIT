# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
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


def get_n_random(low: int, high: int, n: int) -> list[int]:
    """
    Generate a list of n random indices within a given range: low (inclusive) and high (exclusive).
    """
    result = []
    try:
        result = random.sample(range(low, high), n)
    except ValueError:
        logging.getLogger(__name__).debug(f"Sample size of {n} exceeds population size of {high - low}")
    return result
