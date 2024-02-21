# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dotenv
import os

from pyrit.common import path


def load_default_env() -> None:
    """
    Loads an environment file from the given path, or the $PROJECT_ROOT/.env file if no path is given
    Throws an exception if the file is not found
    """
    file_path = path.HOME_PATH / ".env"

    if not file_path.exists():
        raise FileNotFoundError(f"Environment file not found at {file_path}")

    dotenv.load_dotenv(file_path, override=True)


def get_required_value(*, env_var_name: str, passed_value: str) -> str:
    """
    Gets a required value from an environment variable or a passed value,
    prefering the passed value

    If no value is found, raises a KeyError

    :param environment_variable_name: The name of the environment variable
    :type environment_variable_name: str
    :param passed_value: The value passed as an argument
    :type passed_value: str
    :return: The required value
    :rtype: str
    :raises ValueError: If no value is found
    """
    if passed_value:
        return passed_value

    value = os.environ.get(env_var_name)
    if value:
        return value

    raise ValueError(f"Environment variable {env_var_name} is required")
