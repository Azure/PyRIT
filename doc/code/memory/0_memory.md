# Memory

PyRIT's memory component allows users to track and manage a history of interactions throughout an attack scenario. This feature enables the storage, retrieval, and sharing of conversation entries, making it easier to maintain context and continuity in ongoing interactions.

To simplify memory interaction, the `pyrit.memory.CentralMemory` class automatically manages a shared memory instance across all components in a session. Memory must be set explicitly.

**Manual Memory Setting**:

At the beginning of each notebook, make sure to call:
```
# Import initialize_pyrit
# Import the specific constant for the MemoryDatabaseType, or provide the literal value
from pyrit.common import initialize_pyrit, IN_MEMORY, DUCK_DB, AZURE_SQL

initialize_pyrit(memory_db_type: MemoryDatabaseType, memory_instance_kwargs: Optional[Any])
```

The `MemoryDatabaseType` is a `Literal` with 3 options: IN_MEMORY, DUCK_DB, AZURE_SQL. (Read more below)
   - `initialize_pyrit` takes the `MemoryDatabaseType` and an argument list (`memory_instance_kwargs`), to initialize the shared memory instance.

##  Memory Database Type Options

**IN_MEMORY:** _In-Memory DuckDB Database_
   - This option can be preferable if the user does not care about storing conversations or scores in memory beyond the current process. It is used as the default in most of the PyRIT notebooks.
   - **Note**: In in-memory mode, no data is persisted to disk, therefore, all data is lost when the process finishes (from [DuckDB docs](https://duckdb.org/docs/connect/overview.html#in-memory-database))

**DUCK_DB:** _Persistent DuckDB Database_
   - Interactions will be stored in a persistent `DuckDBMemory` instance with a location on-disk. See notebook [here](./1_duck_db_memory.ipynb) for more details.

**AZURE_SQL:** _Azure SQL Database_
   - For examples on setting up `AzureSQLMemory`, please refer to the notebook [here](./7_azure_sql_memory_orchestrators.ipynb).
   - To configure AzureSQLMemory without an extra argument list, these keys should be in your `.env` file:
     - `AZURE_SQL_DB_CONNECTION_STRING`
     - `AZURE_STORAGE_ACCOUNT_DB_DATA_CONTAINER_URL`
