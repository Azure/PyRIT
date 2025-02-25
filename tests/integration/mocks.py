# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator

from sqlalchemy import inspect

from pyrit.memory import DuckDBMemory, MemoryInterface


def get_memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_duckdb_memory()


def get_duckdb_memory() -> Generator[DuckDBMemory, None, None]:
    # Create an in-memory DuckDB engine
    duckdb_memory = DuckDBMemory(db_path=":memory:")

    duckdb_memory.disable_embedding()

    # Reset the database to ensure a clean state
    duckdb_memory.reset_database()
    inspector = inspect(duckdb_memory.engine)

    # Verify that tables are created as expected
    assert "PromptMemoryEntries" in inspector.get_table_names(), "PromptMemoryEntries table not created."
    assert "EmbeddingData" in inspector.get_table_names(), "EmbeddingData table not created."
    assert "ScoreEntries" in inspector.get_table_names(), "ScoreEntries table not created."
    assert "SeedPromptEntries" in inspector.get_table_names(), "SeedPromptEntries table not created."

    yield duckdb_memory
    duckdb_memory.dispose_engine()
