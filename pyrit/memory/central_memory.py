# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional

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
    def get_memory(cls, passed_memory: Optional[MemoryInterface] = None) -> MemoryInterface:
        """
        Returns a centralized memory instance. If `passed_memory` is provided, it's set as the 
        central instance. Otherwise, it checks for Azure SQL configuration and defaults to 
        DuckDBMemory if Azure SQL DB and Azure Storage Account settings are missing.
        """
        if passed_memory:
            cls._memory_instance = passed_memory
            logger.info(f"Using provided memory instance: {type(cls._memory_instance).__name__}")
            return cls._memory_instance

        if cls._memory_instance:
            logger.info(f"Reusing existing memory instance: {type(cls._memory_instance).__name__}")
            return cls._memory_instance

        # Load environment variables
        default_values.load_default_env()

        # Check for Azure SQL settings with get_non_required_value logic
        empty_passed_value = ""
        azure_sql_db_conn_string = default_values.get_non_required_value(
            env_var_name="AZURE_SQL_DB_CONNECTION_STRING",
            passed_value=empty_passed_value
        )
        results_container_url = default_values.get_non_required_value(
            env_var_name="AZURE_STORAGE_ACCOUNT_RESULTS_CONTAINER_URL",
            passed_value=empty_passed_value
        )

        # If both Azure SQL configs are present, use AzureSQLMemory; otherwise, use DuckDBMemory
        if azure_sql_db_conn_string and results_container_url:
            logger.info("Using AzureSQLMemory as central memory.")
            cls._memory_instance = AzureSQLMemory(connection_string=azure_sql_db_conn_string, container_url=results_container_url)
        else:
            logger.info("Using DuckDBMemory due to missing Azure SQL DB configuration.")
            cls._memory_instance = DuckDBMemory()

        return cls._memory_instance