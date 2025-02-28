# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from contextlib import closing
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError

from pyrit.memory import AzureSQLMemory, SeedPromptEntry
from pyrit.memory.memory_models import Base
from pyrit.models import SeedPrompt


def remove_entry(memory: AzureSQLMemory, entry: Base) -> None:  # type: ignore
    """
    Removes an entry from the Table for testing purposes.

    Args:
        entry: An instance of a SQLAlchemy model to be removed from the Table.
    """
    with closing(memory.get_session()) as session:
        try:
            session.delete(entry)
            session.commit()
        except SQLAlchemyError:
            session.rollback()


@pytest.mark.skip(reason="Skipping until #4001 is addressed.")
@pytest.mark.asyncio
async def test_get_seed_prompts_with_metadata_filter(azuresql_instance: AzureSQLMemory):
    value1 = str(uuid4())
    # Integers should work properly as values in the metadata dict
    value2 = np.random.randint(0, 10000)
    sp1 = SeedPrompt(value="sp1", data_type="text", metadata={"key1": value1}, added_by="test")
    sp2 = SeedPrompt(value="sp2", data_type="text", metadata={"key1": value2, "key2": value1}, added_by="test")
    entry1, entry2 = SeedPromptEntry(entry=sp1), SeedPromptEntry(entry=sp2)
    azuresql_instance._insert_entries(entries=[entry1, entry2])

    result = azuresql_instance.get_seed_prompts(metadata={"key1": value1})
    result2 = azuresql_instance.get_seed_prompts(metadata={"key1": value2, "key2": value1})
    assert len(result) == 1
    assert len(result2) == 1
    assert result[0].metadata == {"key1": value1}
    assert result2[0].metadata == {"key1": value2, "key2": value1}

    # Clean up
    remove_entry(azuresql_instance, entry1)
    remove_entry(azuresql_instance, entry2)
    # Ensure that entries are removed
    assert azuresql_instance.get_seed_prompts(metadata={"key1": value1}) == []
    assert azuresql_instance.get_seed_prompts(metadata={"key2": value1}) == []
