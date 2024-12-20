# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def combine_dict(existing_dict: dict[str, str] = None, new_dict: dict[str, str] = None) -> dict[str, str]:
    """
    Combines two dictionaries containing string keys and values into one
    Args:
        existing_dict: Dictionary with existing values
        new_dict: Dictionary with new values to be added to the existing dictionary. 
            Note if there's a key clash, the value in new_dict will be used.
    Returns: combined dictionary
    """
    return {**(existing_dict or {}), **(new_dict or {})}
