# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Literal, get_args

from pyrit.common import default_values
from pyrit.memory import AzureSQLMemory, CentralMemory, DuckDBMemory, MemoryInterface


logger = logging.getLogger(__name__)

MemoryDatabaseType = Literal["InMemory", "DuckDB", "AzureSQL"]


def initialize_pyrit(memory_db_type: MemoryDatabaseType) -> None:
    """
    Initializes PyRIT with the provided memory instance and loads environment files.

    Args:
        memory (MemoryInterface): The memory instance to use for PyRIT.
    """
    if memory_db_type not in get_args(MemoryDatabaseType):
        raise ValueError(
            f"Memory database type '{memory_db_type}' is not a supported type {get_args(MemoryDatabaseType)}"
        )

    default_values.load_environment_files()

    memory: MemoryInterface = None
    if memory_db_type == "InMemory":
        memory = DuckDBMemory(db_path=":memory:")
    elif memory_db_type == "DuckDB":
        memory = DuckDBMemory(db_path=None)
    else:
        memory = AzureSQLMemory()

    CentralMemory.set_memory_instance(memory)
