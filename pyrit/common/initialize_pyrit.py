# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
import logging

from pyrit.common import default_values
from pyrit.memory import AzureSQLMemory, CentralMemory, DuckDBMemory


logger = logging.getLogger(__name__)


class MemoryInstance(Enum):
    """
    An enumeration of the available memory instances for PyRIT.
    """
    IN_MEMORY = 0
    ON_DISK = 1
    AZURE_SQL = 2


def initialize_pyrit(memory_instance: MemoryInstance) -> None:
    """
    Initializes PyRIT with the provided memory instance and loads environment files.

    Args:
        memory (MemoryInterface): The memory instance to use for PyRIT.
    """
    default_values.load_environment_files()

    if memory_instance == MemoryInstance.IN_MEMORY:
        memory = DuckDBMemory(db_path=":memory:")
    elif memory_instance == MemoryInstance.ON_DISK:
        memory = DuckDBMemory(db_path=None)
    else:
        memory = AzureSQLMemory()

    CentralMemory.set_memory_instance(memory)