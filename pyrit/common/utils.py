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


def get_random_indices(low: int, high: int, sample_ratio: float) -> list[int]:
    """
    Generate a list of random indices within a given range based on a sample ratio.

    Args:
        low (int): Lower bound of the range (inclusive).
        high (int): Upper bound of the range (exclusive).
        sample_ratio (float): Ratio of range to sample (0.0 to 1.0).
    """
    if sample_ratio < 0 or sample_ratio > 1:
        raise ValueError("Sample ratio must be between 0 and 1")

    # Special case: return empty list
    if sample_ratio == 0:
        return []

    result = []
    n = math.ceil((high - low) * sample_ratio)

    # Ensure at least 1 index for non-zero sample ratio
    if n == 0:
        n = 1

    try:
        result = random.sample(range(low, high), n)
    except ValueError:
        logger.debug(f"Sample size of {n} exceeds population size of {high - low}")
    return result


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
        regex (str or Pattern): Regular expression pattern to match (for "regex" mode).
        sample_ratio (float): Ratio of words to randomly select (for "random" mode).

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
            sample_ratio = kwargs.get("sample_ratio", 0.5)
            return get_random_indices(0, len(words), sample_ratio)

        case "regex":
            regex = kwargs.get("regex", r".")
            return [i for i, word in enumerate(words) if re.search(regex, word)]

        case "custom":
            custom_indices = kwargs.get("indices", [])
            return [i for i in custom_indices if 0 <= i < len(words)]

        case _:
            return list(range(len(words)))

    return list(range(len(words)))
