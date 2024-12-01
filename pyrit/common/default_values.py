# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dotenv
import os
import logging

from pyrit.common import path


logger = logging.getLogger(__name__)


def load_environment_files() -> None:
    """
    Loads the base environment file from .env if it exists,
    and then loads a single .env.local file if it exists, overriding previous values.
    """
    base_file_path = path.HOME_PATH / ".env"
    local_file_path = path.HOME_PATH / ".env.local"

    # Load the base .env file if it exists
    if base_file_path.exists():
        dotenv.load_dotenv(base_file_path, override=True)
        logger.info(f"Loaded {base_file_path}")
    else:
        dotenv.load_dotenv()

    # Load the .env.local file if it exists, to override base .env values
    if local_file_path.exists():
        dotenv.load_dotenv(local_file_path, override=True)
        logger.info(f"Loaded {local_file_path}")


def get_required_value(*, env_var_name: str, passed_value: str) -> str:
    """
    Gets a required value from an environment variable or a passed value,
    prefering the passed value

    If no value is found, raises a KeyError

    Args:
        env_var_name (str): The name of the environment variable to check
        passed_value (str): The value passed to the function.

    Returns:
        str: The passed value if provided, otherwise the value from the environment variable.

    Raises:
        ValueError: If neither the passed value nor the environment variable is provided.
    """
    if passed_value:
        return passed_value

    value = os.environ.get(env_var_name)
    if value:
        return value

    raise ValueError(f"Environment variable {env_var_name} is required")


def get_non_required_value(*, env_var_name: str, passed_value: str) -> str:
    """
    Gets a non-required value from an environment variable or a passed value,
    prefering the passed value.

    Args:
        env_var_name (str): The name of the environment variable to check.
        passed_value (str): The value passed to the function.

    Returns:
        str: The passed value if provided, otherwise the value from the environment variable.
             If no value is found, returns an empty string.
    """
    if passed_value:
        return passed_value

    value = os.environ.get(env_var_name)
    if value:
        return value

    return ""
