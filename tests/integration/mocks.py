# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator

from sqlalchemy import inspect

from pyrit.memory import MemoryInterface, SQLiteMemory


def get_memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_sqlite_memory()


def get_sqlite_memory() -> Generator[SQLiteMemory, None, None]:
    # Create an in-memory SQLite engine
    sqlite_memory = SQLiteMemory(db_path=":memory:")

    sqlite_memory.disable_embedding()

    # Reset the database to ensure a clean state
    sqlite_memory.reset_database()
    inspector = inspect(sqlite_memory.engine)

    # Verify that tables are created as expected
    assert "PromptMemoryEntries" in inspector.get_table_names(), "PromptMemoryEntries table not created."
    assert "EmbeddingData" in inspector.get_table_names(), "EmbeddingData table not created."
    assert "ScoreEntries" in inspector.get_table_names(), "ScoreEntries table not created."
    assert "SeedPromptEntries" in inspector.get_table_names(), "SeedPromptEntries table not created."

    yield sqlite_memory
    sqlite_memory.dispose_engine()
