# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Generator
from unittest.mock import patch

import pytest
from sqlalchemy import inspect

# This limits retries and speeds up execution
# note this needs to be set before libraries that use them are imported

# Note this module needs to be imported at the top of a file so the values are modified

os.environ["RETRY_MAX_NUM_ATTEMPTS"] = "2"
os.environ["RETRY_WAIT_MIN_SECONDS"] = "0"
os.environ["RETRY_WAIT_MAX_SECONDS"] = "1"


from pyrit.memory.central_memory import CentralMemory  # noqa: E402
from pyrit.memory.duckdb_memory import DuckDBMemory  # noqa: E402


@pytest.fixture
def duckdb_instance() -> Generator[DuckDBMemory, None, None]:
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


@pytest.fixture()
def patch_central_database(duckdb_instance):
    """Fixture to mock CentralMemory.get_memory_instance"""
    with patch.object(CentralMemory, "get_memory_instance", return_value=duckdb_instance) as duck_db_memory:
        yield duck_db_memory
