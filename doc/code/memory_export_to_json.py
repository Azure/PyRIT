# %% [markdown]
# The `pyrit.memory.data_exporter` module provides functionality to dump the database tables into JSON files, creating one file per table. In a nutshell, this can be used as follows

# %%
# Export db tables to JSON
from pyrit.memory import DuckDBMemory
from pyrit.memory.memory_exporter import MemoryExporter

duckdb_memory = DuckDBMemory()
data_exporter = MemoryExporter(duckdb_memory)
data_exporter.export_all_tables()

# %%
# Cleanup DuckDB resources
duckdb_memory.dispose_engine()

# %%
