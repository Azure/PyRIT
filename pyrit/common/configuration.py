# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dotenv
import os

def get_env_variable(key: str, default: str = None) -> str:
    """
    Get the value of an environment variable or return a default value

    Args:
        key: The environment variable to get
        default: The default value to return if the environment variable is not set

    Returns:
        The value of the environment variable or the default value
    """
    dotenv.load_dotenv()

    return os.environ.get(key, default)
