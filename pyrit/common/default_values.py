# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dotenv
import os


from pyrit.common import path


def load_default_env() -> None:
    """
    Loads an environment file from the $PROJECT_ROOT/.env file if it exists,
    or if not, loads from the default dotenv .env file
    """
    file_path = path.HOME_PATH / ".env"

    if not file_path.exists():
        dotenv.load_dotenv()
        return

    dotenv.load_dotenv(file_path, override=True)


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
