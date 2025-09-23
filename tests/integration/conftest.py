# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
from typing import Generator

import pytest
from sqlalchemy import inspect

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.memory.azure_sql_memory import AzureSQLMemory
from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.sqlite_memory import SQLiteMemory

# This limits retries to 10 attempts with a 1 second wait between retries
# note this needs to be set before libraries that use them are imported

# Note this module needs to be imported at the top of a file so the values are modified

os.environ["RETRY_MAX_NUM_ATTEMPTS"] = "9"
os.environ["RETRY_WAIT_MIN_SECONDS"] = "0"
os.environ["RETRY_WAIT_MAX_SECONDS"] = "1"

initialize_pyrit(memory_db_type=IN_MEMORY)


@pytest.fixture
def azuresql_instance() -> Generator[AzureSQLMemory, None, None]:
    azuresql_memory = AzureSQLMemory()
    azuresql_memory.disable_embedding()

    inspector = inspect(azuresql_memory.engine)

    # Verify that tables are created as expected
    assert "PromptMemoryEntries" in inspector.get_table_names(), "PromptMemoryEntries table not created."
    assert "EmbeddingData" in inspector.get_table_names(), "EmbeddingData table not created."
    assert "ScoreEntries" in inspector.get_table_names(), "ScoreEntries table not created."
    assert "SeedPromptEntries" in inspector.get_table_names(), "SeedPromptEntries table not created."

    CentralMemory.set_memory_instance(azuresql_memory)
    yield azuresql_memory
    azuresql_memory.dispose_engine()


def pytest_configure(config):
    # Let pytest know about your custom marker for help/usage info
    config.addinivalue_line("markers", "run_only_if_all_tests: skip test unless RUN_ALL_TESTS is set to true")


def pytest_collection_modifyitems(config, items):
    run_all = os.getenv("RUN_ALL_TESTS", "").lower() == "true"
    skip_marker = pytest.mark.skip(reason="RUN_ALL_TESTS is not set to true")
    for item in items:
        if "run_only_if_all_tests" in item.keywords and not run_all:
            item.add_marker(skip_marker)


@pytest.fixture
def sqlite_instance() -> Generator[SQLiteMemory, None, None]:
    # Create an in-memory SQLite engine
    sqlite_memory = SQLiteMemory(db_path=":memory:")
    temp_dir = tempfile.TemporaryDirectory()
    sqlite_memory.results_path = temp_dir.name

    sqlite_memory.disable_embedding()

    # Reset the database to ensure a clean state
    sqlite_memory.reset_database()
    inspector = inspect(sqlite_memory.engine)

    # Verify that tables are created as expected
    assert "PromptMemoryEntries" in inspector.get_table_names(), "PromptMemoryEntries table not created."
    assert "EmbeddingData" in inspector.get_table_names(), "EmbeddingData table not created."
    assert "ScoreEntries" in inspector.get_table_names(), "ScoreEntries table not created."
    assert "SeedPromptEntries" in inspector.get_table_names(), "SeedPromptEntries table not created."

    CentralMemory.set_memory_instance(sqlite_memory)
    yield sqlite_memory
    temp_dir.cleanup()
    sqlite_memory.dispose_engine()
