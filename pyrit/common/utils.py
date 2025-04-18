# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import random
import re
from typing import List, Literal, Union

logger = logging.getLogger(__name__)


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


def get_random_indices(start: int, size: int, percentage: int) -> List[int]:
    """
    Generate a list of random indices based on a specified percentage of the total size.
    The indices are selected from the range [start, start + size).

    Args:
        start (int): Starting index (inclusive). It's the first index that could possibly be selected.
        size (int): Size of the collection to select from. This is the total number of indices available.
            For example, if `start` is 0 and `size` is 10, the available indices are [0, 1, 2, ..., 9].
        percentage (int): Percentage of indices to select from the specified range [0 to 100].
            For example, 30 would mean 30% of the total size, and 50 would mean half of the total size.
    """
    if start < 0:
        raise ValueError("Start index must be non-negative")
    if size <= 0:
        raise ValueError("Size must be greater than 0")
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")

    if percentage == 0:
        return []
    if percentage == 100:
        return list(range(start, start + size))

    # Convert percentage to proportion
    sample_proportion = percentage / 100.0

    n = max(math.ceil(size * sample_proportion), 1)  # the number of indices to select

    return random.sample(range(start, start + size), n)


def select_word_indices(
    words: List[str], mode: Literal["all", "custom", "keywords", "random", "regex"], **kwargs
) -> List[int]:
    """
    Select indices from a list of words based on specified selection mode.

    Supported modes:
        - "all": Select all word indices.
        - "custom": Select custom indices.
        - "keywords": Select indices of specific keywords.
        - "random": Select random indices based on a sample ratio.
        - "regex": Select indices matching a regular expression.

    Args:
        words (List[str]): A list of words to select from.
        mode (str, optional): Selection mode. Defaults to "all".

    Keyword Arguments:
        indices (List[int]): Custom indices to select (for "custom" mode).
        keywords (List[str]): List of keywords to match (for "keywords" mode).
        percentage (int): Percentage of indices to select (for "random" mode).
        regex (str or Pattern): Regular expression pattern to match (for "regex" mode).

    Returns:
        List[int]: Indices of selected words.
    """
    if not words:
        return []

    if mode not in ["all", "keywords", "random", "regex", "custom"]:
        logger.warning(f"Unsupported word selection mode '{mode}'. Defaulting to 'all'.")
        mode = "all"

    match mode:
        case "all":
            return list(range(len(words)))

        case "keywords":
            word_list = kwargs.get("keywords", [])
            return [i for i, word in enumerate(words) if word in word_list]

        case "random":
            percentage = kwargs.get("percentage", 50)
            return get_random_indices(0, len(words), percentage)

        case "regex":
            regex = kwargs.get("regex", r".")
            return [i for i, word in enumerate(words) if re.search(regex, word)]

        case "custom":
            custom_indices = kwargs.get("indices", [])
            return [i for i in custom_indices if 0 <= i < len(words)]
