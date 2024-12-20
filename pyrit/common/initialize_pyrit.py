# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
import logging

from pyrit.common import default_values
from pyrit.memory import AzureSQLMemory, CentralMemory, DuckDBMemory, MemoryInterface


logger = logging.getLogger(__name__)


class MemoryInstance(Enum):
    """
    An enumeration of the available memory instances for PyRIT.
    """
    IN_MEMORY = DuckDBMemory(db_path=":memory:")
    ON_DISK = DuckDBMemory()
    AZURE_SQL = AzureSQLMemory()


def initialize_pyrit(memory_instance: MemoryInterface) -> None:
    """
    Initializes PyRIT with the provided memory instance and loads environment files.

    Args:
        memory (MemoryInterface): The memory instance to use for PyRIT.
    """
    default_values.load_environment_files()

    CentralMemory.set_memory_instance(memory_instance.value)