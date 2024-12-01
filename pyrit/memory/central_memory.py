# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.common import default_values
from pyrit.memory import AzureSQLMemory
from pyrit.memory import DuckDBMemory
from pyrit.memory.memory_interface import MemoryInterface

logger = logging.getLogger(__name__)


class CentralMemory:
    """
    Provides a centralized memory instance across the framework. If a memory instance is passed,
    it will be reused for future calls. Otherwise, it uses AzureSQLMemory if configuration values
    are found, defaulting to DuckDBMemory if not.
    """

    _memory_instance: MemoryInterface = None

    @classmethod
    def set_memory_instance(cls, passed_memory: MemoryInterface) -> None:
        """
        Set a provided memory instance as the central instance for subsequent calls.

        Args:
            passed_memory (MemoryInterface): The memory instance to set as the central instance.
        """
        cls._memory_instance = passed_memory
        logger.info(f"Central memory instance set to: {type(cls._memory_instance).__name__}")

    @classmethod
    def get_memory_instance(cls) -> MemoryInterface:
        """
        Returns a centralized memory instance. Initializes it to AzureSQLMemory if
        Azure SQL/Storage Account configuration values are found, otherwise defaults
        to DuckDBMemory.
        """
        if cls._memory_instance:
            logger.info(f"Reusing existing memory instance: {type(cls._memory_instance).__name__}")
            return cls._memory_instance

        # Check for Azure SQL settings
        empty_passed_value = ""
        azure_sql_db_conn_string = default_values.get_non_required_value(
            env_var_name="AZURE_SQL_DB_CONNECTION_STRING", passed_value=empty_passed_value
        )
        results_container_url = default_values.get_non_required_value(
            env_var_name="AZURE_STORAGE_ACCOUNT_RESULTS_CONTAINER_URL", passed_value=empty_passed_value
        )

        # If both Azure SQL configs are present, use AzureSQLMemory; otherwise, use DuckDBMemory
        if azure_sql_db_conn_string and results_container_url:
            logger.info("Using AzureSQLMemory as central memory.")
            cls._memory_instance = AzureSQLMemory(
                connection_string=azure_sql_db_conn_string, container_url=results_container_url
            )
        else:
            logger.info("Using DuckDBMemory due to missing Azure SQL DB configuration.")
            cls._memory_instance = DuckDBMemory()

        return cls._memory_instance
