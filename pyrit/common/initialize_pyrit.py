# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import dotenv
import logging
from typing import Literal, Optional, Any, get_args

from pyrit.common import path
from pyrit.memory import AzureSQLMemory, CentralMemory, DuckDBMemory, MemoryInterface


logger = logging.getLogger(__name__)

IN_MEMORY = Literal["InMemory"]
DUCK_DB = Literal["DuckDB"]
AZURE_SQL = Literal["AzureSQL"]
MemoryDatabaseType = Literal["InMemory", "DuckDB", "AzureSQL"]


def _load_environment_files() -> None:
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


def initialize_pyrit(memory_db_type: MemoryDatabaseType, **memory_instance_kwargs: Optional[Any]) -> None:
    """
    Initializes PyRIT with the provided memory instance and loads environment files.

    Args:
        memory_db_type (MemoryDatabaseType): The MemoryDatabaseType string literal which indicates the memory
            instance to use for central memory. Options include "InMemory", "DuckDB", and "AzureSQL".
        **memory_instance_kwargs (Optional[Any]): Additional keyword arguments to pass to the memory instance.
    """
    if memory_db_type not in get_args(MemoryDatabaseType):
        raise ValueError(
            f"Memory database type '{memory_db_type}' is not a supported type {get_args(MemoryDatabaseType)}"
        )

    _load_environment_files()

    memory: MemoryInterface = None
    if memory_db_type == IN_MEMORY:
        logger.info("Using in-memory DuckDB database.")
        memory = DuckDBMemory(db_path=":memory:", **memory_instance_kwargs)
    elif memory_db_type == DUCK_DB:
        logger.info("Using persistent DuckDB database.")
        memory = DuckDBMemory(**memory_instance_kwargs)
    else:
        logger.info("Using AzureSQL database.")
        memory = AzureSQLMemory(**memory_instance_kwargs)

    CentralMemory.set_memory_instance(memory)
