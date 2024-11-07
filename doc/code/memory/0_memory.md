# Memory

PyRIT's memory component enables users to maintain a history of interactions during the course of an attack. At its core, this feature allows for the storage, retrieval, and sharing of conversation entries.

To make memory interaction simple and easy to use, the `prit.memory.CentralMemory` class allows you to set the memory instance which will be used across all components in a session.

- **Set Memory Manually**: Use `CentralMemory.set_memory_instance(passed_memory)` to specify the memory instance explicitly.
- **Automatic Memory Detection**:
  - **Azure SQL DB**: If no memory instance is explicitly set, `CentralMemory` will check the `.env` file for Azure SQL settings. If found, it automatically sets the memory instance to `AzureSQLMemory`, storing results in the Azure SQL Database.
  - **Local DuckDB**: If Azure SQL settings are not configured, `CentralMemory` defaults to using `DuckDBMemory`, storing results locally in DuckDB.
