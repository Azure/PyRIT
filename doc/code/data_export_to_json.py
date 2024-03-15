# %% [markdown]
# The `pyrit.memory.data_exporter` module provides functionality to dump the database tables into JSON files, creating one file per table. In a nutshell, this can be used as follows

# %%
# Export db tables to JSON
from pyrit.memory import DuckDBMemory
from pyrit.memory.data_exporter import DataExporter
duckdb_memory = DuckDBMemory()
data_exporter = DataExporter(duckdb_memory)
data_exporter.export_all_tables()

# %%
# Cleanup DuckDB resources
duckdb_memory.dispose_engine()

# %%



