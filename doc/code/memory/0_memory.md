# Memory

PyRIT's memory component allows users to track and manage a history of interactions throughout an attack scenario. This feature enables the storage, retrieval, and sharing of conversation entries, making it easier to maintain context and continuity in ongoing interactions.

To simplify memory interaction, the `pyrit.memory.CentralMemory` class automatically manages a shared memory instance across all components in a session. Memory must be set explicitly.

**Manual Memory Setting**:

At the beginning of each notebook, make sure to call:
```
from pyrit.common.initialize_pyrit import initialize_pyrit

# MemoryDatabaseType = Literal["InMemory", "DuckDB", "AzureSQL"]
initialize_pyrit(memory_db_type: MemoryDatabaseType, memory_instance_kwargs: Optional[Any])
```

The `MemoryDatabaseType` is a `Literal` with 3 options: "InMemory", "DuckDB", "AzureSQL". (Read more below)
   - `initialize_pyrit` takes the `MemoryDatabaseType` and an argument list (`memory_instance_kwargs`), to pass to the memory instance.
   - When called, `CentralMemory.set_memory_instance(passed_memory)` explicitly defines the memory instance to be used.

###  Memory Database Type Options

**DuckDB:** _Local DuckDB_
   - Interactions will be stored in a local `DuckDBMemory` instance. See notebook [here](./1_duck_db_memory.ipynb).

**AzureSQL:** _Azure SQL Database_
   - For examples on setting up `AzureSQLMemory`, please refer to the notebook [here](./7_azure_sql_memory_orchestrators.ipynb).
   - To configure AzureSQLMemory without an extra arguemnt list, these keys should be in your `.env` file:
     - `AZURE_SQL_DB_CONNECTION_STRING`
     - `AZURE_STORAGE_ACCOUNT_RESULTS_CONTAINER_URL`

**InMemory:** _Duck DB with db_path=":memory:"_
   - This option is sometimes preferable when using PyRIT for quick interactions that do not require storing results in a database.
   - **Note**: This option will not work for any functionality that checks interactions from memory, such as orchestrators.
