# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any, Literal, Optional, Union, get_args

import dotenv

from pyrit.common import path
from pyrit.memory import (
    AzureSQLMemory,
    CentralMemory,
    MemoryInterface,
    SQLiteMemory,
)

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


def initialize_pyrit(memory_db_type: Union[MemoryDatabaseType, str], **memory_instance_kwargs: Optional[Any]) -> None:
    """
    Initializes PyRIT with the provided memory instance and loads environment files.

    Args:
        memory_db_type (MemoryDatabaseType): The MemoryDatabaseType string literal which indicates the memory
            instance to use for central memory. Options include "InMemory", "SQLite", and "AzureSQL".
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

    memory: MemoryInterface = None

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
