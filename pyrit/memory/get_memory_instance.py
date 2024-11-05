# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.common import default_values
from pyrit.memory import AzureSQLMemory
from pyrit.memory import DuckDBMemory
from pyrit.memory.memory_interface import MemoryInterface

logger = logging.getLogger(__name__)


def get_memory_instance() -> MemoryInterface:
    """
    Returns a memory instance based on the presence of AZURE_SQL_DB_CONNECTION_STRING and
    AZURE_STORAGE_ACCOUNT_RESULTS_CONTAINER_URL.
    If these variables exists, it uses `AzureSQLMemory`; otherwise, defaults to `DuckDBMemory`.
    """
    # Load environment variables
    default_values.load_default_env()

    # Check if Azure SQL connection string is provided
    empty_passed_value = ""
    azure_sql_db_conn_string = default_values.get_non_required_value(
            env_var_name="AZURE_SQL_DB_CONNECTION_STRING", passed_value=empty_passed_value
        )
    results_container_url = default_values.get_non_required_value(
            env_var_name="AZURE_STORAGE_ACCOUNT_RESULTS_CONTAINER_URL", passed_value=empty_passed_value)
    if azure_sql_db_conn_string and results_container_url: 
        # If both values are available, initialize AzureSQLMemory
        logger.info("Using AzureSQLMemory.")
        return AzureSQLMemory(connection_string=azure_sql_db_conn_string, container_url=results_container_url)
    else:
        # If required Azure SQL values are missing, fallback to DuckDBMemory
        logger.info(f"Using DuckDBMemory due to missing Azure SQL DB configuration")
        return DuckDBMemory()