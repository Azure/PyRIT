# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from tempfile import NamedTemporaryFile
import pytest

from sqlalchemy import inspect

from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.prompt_target import TextTarget


@pytest.fixture
def memory() -> MemoryInterface:  # type: ignore
    # Create an in-memory DuckDB engine
    duckdb_memory = DuckDBMemory(db_path=":memory:")

    # Reset the database to ensure a clean state
    duckdb_memory.reset_database()
    inspector = inspect(duckdb_memory.engine)

    # Verify that tables are created as expected
    assert "ConversationStore" in inspector.get_table_names(), "ConversationStore table not created."
    assert "EmbeddingStore" in inspector.get_table_names(), "EmbeddingStore table not created."

    yield duckdb_memory
    duckdb_memory.dispose_engine()


def test_send_prompt_user_no_system(memory: DuckDBMemory):
    no_op = TextTarget(memory=memory)

    no_op.send_prompt(
        normalized_prompt="hi, I am a victim chatbot, how can I help?", conversation_id="1", normalizer_id="2"
    )

    chats = no_op.memory.get_memories_with_conversation_id(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "user"


def test_send_prompt_stream(memory: DuckDBMemory):
    with NamedTemporaryFile(mode="w+", delete=False) as tmp_file:
        prompt = "hi, I am a victim chatbot, how can I help?"

        no_op = TextTarget(memory=memory, text_stream=tmp_file)
        no_op.send_prompt(normalized_prompt=prompt, conversation_id="1", normalizer_id="2")

        tmp_file.seek(0)
        content = tmp_file.read()

    os.remove(tmp_file.name)

    assert prompt in content, "The prompt was not found in the temporary file content."
