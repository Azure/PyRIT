# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def combine_dict(dict1: dict[str, str] = None, dict2: dict[str, str]= None) -> dict[str, str]:
    """
    Combines two dictionaries containing string keys and values into one
    Args:  
        dict1: Dictionary 1
        dict2: Dictionary 2
    Returns: combined dictionary
    """
    combined_dict = dict1.copy() if dict1 else {}
    if dict2: 
        for key, value in dict2.items():
            if dict1:
                combined_dict[key] = value
            else:  # if labels is None
                combined_dict = {key: value}
    return combined_dict
