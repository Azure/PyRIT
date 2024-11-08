# Memory

PyRIT's memory component allows users to track and manage a history of interactions throughout an attack scenario. This feature enables the storage, retrieval, and sharing of conversation entries, making it easier to maintain context and continuity in ongoing interactions.

To simplify memory interaction, the `prit.memory.CentralMemory` class automatically manages a shared memory instance across all components in a session. Memory is selected based on the following priority order:

1. **Manual Memory Setting (Highest Priority)**:
   - If set, `CentralMemory.set_memory_instance(passed_memory)` explicitly defines the memory instance to be used, overriding all other settings.
   - For examples on setting up `AzureSQLMemory`, please refer to the notebook [here](./7_azure_sql_memory_orchestrators.ipynb).

2. **Azure SQL Database**:
   - If no manual instance is provided, `CentralMemory` will check for Azure SQL settings in the `.env` file or environment. If the following variables are detected, `CentralMemory` automatically configures `AzureSQLMemory` for storage in an Azure SQL Database:
     - `AZURE_SQL_DB_CONNECTION_STRING`
     - `AZURE_STORAGE_ACCOUNT_RESULTS_CONTAINER_URL`

3. **Local DuckDB (Default)**:
   - If neither a manual memory instance nor Azure SQL settings are available, `CentralMemory` defaults to `DuckDBMemory`, storing interactions locally in a DuckDB instance.
