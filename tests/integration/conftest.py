# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Generator

import pytest
from sqlalchemy import inspect

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.memory.azure_sql_memory import AzureSQLMemory
from pyrit.memory.central_memory import CentralMemory

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
