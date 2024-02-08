# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os


def get_required_value(environment_variable_name: str, passed_value: str) -> str:
    """
    Gets a required value from an environment variable or a passed value,
    prefering the passed value

    If no value is found, raises a KeyError
    """
    if passed_value:
        return passed_value

    value = os.environ.get(environment_variable_name)
    if value:
        return value

    raise ValueError(f"Environment variable {environment_variable_name} is required")
