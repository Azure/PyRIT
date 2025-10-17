# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import pathlib
import sys
from typing import Any, Literal, Optional, Sequence, Union, get_args

import dotenv

from pyrit.common import path
from pyrit.memory import (
    AzureSQLMemory,
    CentralMemory,
    MemoryInterface,
    SQLiteMemory,
)
from pyrit.setup.pyrit_default_value import reset_default_values

logger = logging.getLogger(__name__)

IN_MEMORY = "InMemory"
SQLITE = "SQLite"
AZURE_SQL = "AzureSQL"
MemoryDatabaseType = Literal["InMemory", "SQLite", "AzureSQL"]


def _load_environment_files() -> None:
    """
    Loads the base environment file from .env if it exists,
    and then loads a single .env.local file if it exists, overriding previous values.
    """
    base_file_path = path.HOME_PATH / ".env"
    local_file_path = path.HOME_PATH / ".env.local"

    # Load the base .env file if it exists
    if base_file_path.exists():
        dotenv.load_dotenv(base_file_path, override=True, interpolate=True)
        logger.info(f"Loaded {base_file_path}")
    else:
        dotenv.load_dotenv(verbose=True)

    # Load the .env.local file if it exists, to override base .env values
    if local_file_path.exists():
        dotenv.load_dotenv(local_file_path, override=True, interpolate=True)
        logger.info(f"Loaded {local_file_path}")
    else:
        dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv(".env.local"), override=True, verbose=True)


def _execute_initialization_scripts(*, script_paths: Sequence[Union[str, pathlib.Path]]) -> None:
    """
    Executes Python initialization scripts in order.

    These scripts are executed to configure default values and set up global variables.
    Scripts should use explicit functions to set global variables rather than relying
    on naming conventions:

    - Use set_global_variable(name="var_name", value=var_value) to set global variables
    - Use set_default_value() to configure class parameter defaults

    Args:
        script_paths (Sequence[Union[str, pathlib.Path]]): Sequence of file paths to Python scripts to execute.

    Raises:
        FileNotFoundError: If a script path does not exist.
        ValueError: If a script path is not a Python file.

    Example:
        Script content (my_init.py):
            # Helper variables (local to the script)
            _temp_config = {"key": "value"}
            
            # Explicitly set global variables
            from pyrit.setup import set_global_variable, set_default_value
            set_global_variable(name="myVar", value="test_value")

            # Configure default values
            from pyrit.prompt_target import OpenAIChatTarget
            set_default_value(class_type=OpenAIChatTarget, parameter_name="temperature", value=0.7)

        Usage:
            initialize_pyrit(memory_db_type="InMemory", initialization_scripts=["my_init.py"])
            print(myVar)  # "test_value" is accessible via explicit set_global_variable call
    """

    # Get the __main__ module's globals to inject variables into the caller's namespace
    main_globals = sys.modules["__main__"].__dict__

    for script_path in script_paths:
        # Convert to Path object if string
        script = pathlib.Path(script_path)

        # Validate the script exists
        if not script.exists():
            raise FileNotFoundError(f"Initialization script not found: {script}")

        # Validate it's a Python file
        if script.suffix != ".py":
            raise ValueError(f"Initialization script must be a Python file (.py): {script}")

        logger.info(f"Executing initialization script: {script}")

        # Read and execute the script in a temporary namespace
        try:
            with open(script, "r", encoding="utf-8") as f:
                script_content = f.read()

            # Create a temporary namespace that includes main_globals for imports and references
            # but allows us to track what variables were newly defined by this script
            script_globals = main_globals.copy()

            # Track variables that existed before script execution
            pre_execution_vars = set(script_globals.keys())

            # Execute the script in the temporary global namespace
            exec(script_content, script_globals)

            # Identify new variables added by the script
            post_execution_vars = set(script_globals.keys())
            new_vars = post_execution_vars - pre_execution_vars

            logger.debug(f"Initialization script {script.name} executed successfully. ")

        except Exception as e:
            logger.error(f"Error executing initialization script {script}: {e}")
            raise


def initialize_pyrit(
    memory_db_type: Union[MemoryDatabaseType, str],
    *,
    initialization_scripts: Optional[Sequence[Union[str, pathlib.Path]]] = None,
    **memory_instance_kwargs: Optional[Any],
) -> None:
    """
    Initializes PyRIT with the provided memory instance and loads environment files.

    Args:
        memory_db_type (MemoryDatabaseType): The MemoryDatabaseType string literal which indicates the memory
            instance to use for central memory. Options include "InMemory", "SQLite", and "AzureSQL".
        initialization_scripts (Optional[Sequence[Union[str, pathlib.Path]]]): Optional sequence of Python script paths
            to execute in order. These scripts can set global variables and configure default values using
            set_default_value from pyrit.setup.
        **memory_instance_kwargs (Optional[Any]): Additional keyword arguments to pass to the memory instance.
    """
    # Handle DuckDB deprecation before validation
    if memory_db_type == "DuckDB":
        logger.warning(
            "DuckDB is no longer supported and has been replaced by SQLite for better compatibility and performance. "
            "Please update your code to use SQLite instead. "
            "For migration guidance, see the SQLite Memory documentation at: "
            "doc/code/memory/1_sqlite_memory.ipynb. "
            "Using in-memory SQLite instead."
        )
        memory_db_type = IN_MEMORY

    _load_environment_files()

    # Reset all default values before executing initialization scripts
    # This ensures a clean state for each initialization
    reset_default_values()

    # Set up memory BEFORE executing initialization scripts
    # This is critical because initialization scripts may instantiate objects
    # (like prompt targets) that require central memory to be initialized
    memory: MemoryInterface

    if memory_db_type == IN_MEMORY:
        logger.info("Using in-memory SQLite database.")
        memory = SQLiteMemory(db_path=":memory:", **memory_instance_kwargs)
    elif memory_db_type == SQLITE:
        logger.info("Using persistent SQLite database.")
        memory = SQLiteMemory(**memory_instance_kwargs)
    elif memory_db_type == AZURE_SQL:
        logger.info("Using AzureSQL database.")
        memory = AzureSQLMemory(**memory_instance_kwargs)
    else:
        raise ValueError(
            f"Memory database type '{memory_db_type}' is not a supported type {get_args(MemoryDatabaseType)}"
        )
    CentralMemory.set_memory_instance(memory)

    # Execute initialization scripts AFTER memory is set up
    if initialization_scripts:
        _execute_initialization_scripts(script_paths=initialization_scripts)
