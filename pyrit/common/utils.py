# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import random
import re

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


def get_random_indices(low: int, high: int, sample_ratio: float) -> list[int]:
    """
    Generate a list of random indices within a given range based on a sample ratio.
    Args:
        low: Lower bound of the range (inclusive).
        high: Upper bound of the range (exclusive).
        sample_ratio: Ratio of range to sample (0.0 to 1.0).
    """
    # Special case: return empty list
    if sample_ratio == 0:
        return []

    result = []
    n = math.ceil((high - low) * sample_ratio)

    # Ensure at least 1 index for non-zero sample ratio
    if sample_ratio > 0 and n == 0:
        n = 1

    try:
        result = random.sample(range(low, high), n)
    except ValueError:
        logging.getLogger(__name__).debug(f"Sample size of {n} exceeds population size of {high - low}")
    return result


def select_word_indices(words: List[str], mode: str = "all", **kwargs):
    """
    Select indices from a list of words based on specified selection mode.

    Args:
        words (list): A list of words to select from.
        mode (str, optional): Selection mode.
            Supported modes:
                - "all": Select all word indices,.
                - "regex": Select indices matching a regular expression.
                - "keywords": Select indices of specific keywords.
                - "random": Select random indices based on a sample ratio.

    Returns:
        list: Indices of selected words.
    """
    if mode == "all":
        return list(range(len(words)))

    elif mode == "regex":
        regex = kwargs.get("regex", r".")
        return [i for i, word in enumerate(words) if re.search(regex, word)]

    elif mode == "keywords":
        word_list = kwargs.get("keywords", [])
        return [i for i, word in enumerate(words) if word in word_list]

    elif mode == "random":
        sample_ratio = kwargs.get("sample_ratio", 0.5)
        return get_random_indices(0, len(words), sample_ratio)

    # TODO: add more modes here ...

    return list(range(len(words)))
