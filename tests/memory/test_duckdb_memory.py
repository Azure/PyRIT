# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pytest
import uuid
import datetime
from unittest.mock import MagicMock

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect
from sqlalchemy import String, DateTime, Float, Enum, JSON, ForeignKey, Index, INTEGER, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql.sqltypes import NullType
from sqlalchemy.types import String, DateTime

from pyrit.memory.memory_models import PromptMemoryEntry, EmbeddingData, PromptDataType
from pyrit.memory import DuckDBMemory
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_converter.base64_converter import Base64Converter
from pyrit.prompt_converter.prompt_converter import PromptConverterList
from pyrit.prompt_target.text_target import TextTarget


@pytest.fixture
def setup_duckdb_database():
    # Create an in-memory DuckDB engine
    duckdb_memory = DuckDBMemory(db_path=":memory:")

    # Reset the database to ensure a clean state
    duckdb_memory.reset_database()
    inspector = inspect(duckdb_memory.engine)

    # Verify that tables are created as expected
    assert "PromptMemoryEntries" in inspector.get_table_names(), "PromptMemoryEntries table not created."
    assert "EmbeddingData" in inspector.get_table_names(), "EmbeddingData table not created."

    yield duckdb_memory
    duckdb_memory.dispose_engine()


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.commit = MagicMock()
    session.query = MagicMock()
    session.merge = MagicMock()
    session.rollback = MagicMock()
    return session


def test_conversation_data_schema(setup_duckdb_database):
    inspector = inspect(setup_duckdb_database.engine)
    columns = inspector.get_columns("PromptMemoryEntries")
    column_names = [col["name"] for col in columns]

    # Expected columns in ConversationData
    expected_columns = ["id",
                        "role",
                        "conversation_id",
                        "sequence",
                        "timestamp",
                        "labels",
                        "prompt_metadata",
                        "converters",
                        "prompt_target",
                        "original_prompt_data_type",
                        "original_prompt_text",
                        "original_prompt_data_sha256",
                        "converted_prompt_data_type",
                        "converted_prompt_text",
                        "converted_prompt_data_sha256"]

    for column in expected_columns:
        assert column in column_names, f"{column} not found in PromptMemoryEntries schema."


def test_embedding_data_schema(setup_duckdb_database):
    inspector = inspect(setup_duckdb_database.engine)
    columns = inspector.get_columns("EmbeddingData")
    column_names = [col["name"] for col in columns]

    # Expected columns in EmbeddingData
    expected_columns = ["uuid", "embedding", "embedding_type_name"]
    for column in expected_columns:
        assert column in column_names, f"{column} not found in EmbeddingData schema."


def test_conversation_data_column_types(setup_duckdb_database):
    inspector = inspect(setup_duckdb_database.engine)
    columns = inspector.get_columns("PromptMemoryEntries")
    column_types = {col["name"]: type(col["type"]) for col in columns}

    # Expected column types in ConversationData
    expected_column_types = {
        "id": UUID,
        "role": String,
        "conversation_id": String,
        "sequence": INTEGER,
        "timestamp": DateTime,
        "labels": String,
        "prompt_metadata": String,
        "converters": String,
        "prompt_target": String,
        "original_prompt_data_type": String,
        "original_prompt_text": String,
        "original_prompt_data_sha256": String,
        "converted_prompt_data_type": String,
        "converted_prompt_text": String,
        "converted_prompt_data_sha256": String,
    }

    for column, expected_type in expected_column_types.items():
        if column != "labels":
            assert column in column_types, f"{column} not found in PromptMemoryEntries schema."
            assert issubclass(
                column_types[column], expected_type
            ), f"Expected {column} to be a subclass of {expected_type}, got {column_types[column]} instead."



def test_embedding_data_column_types(setup_duckdb_database):
    inspector = inspect(setup_duckdb_database.engine)
    columns = inspector.get_columns("EmbeddingData")
    column_types = {col["name"]: col["type"].__class__ for col in columns}

    # Expected column types in EmbeddingData
    expected_column_types = {
        "id": UUID,
        "embedding": ARRAY,
        "embedding_type_name": String,
    }

    for column, expected_type in expected_column_types.items():
        if column != "embedding":
            assert column in column_types, f"{column} not found in EmbeddingStore schema."
            # Allow for flexibility in type representation (String vs. VARCHAR)
            assert issubclass(
                column_types[column], expected_type
            ), f"Expected {column} to be a subclass of {expected_type}, got {column_types[column]} instead."
    # Handle 'embedding' column separately
    assert "embedding" in column_types, "'embedding' column not found in EmbeddingData schema."
    # Check if 'embedding' column type is either NullType (due to reflection issue) or ARRAY
    assert column_types["embedding"] in [
        NullType,
        ARRAY,
    ], f"Unexpected type for 'embedding' column: {column_types['embedding']}"


def test_insert_entry(setup_duckdb_database):
    session = setup_duckdb_database.get_session()
    entry = PromptMemoryEntry(
        id=uuid.uuid4(),
        conversation_id="123",
        role="user",

        original_prompt_data_type=PromptDataType.TEXT,
        original_prompt_text="Hello",
        original_prompt_data_sha256="abc",
    )
    # Use the insert_entry method to insert the entry into the database
    setup_duckdb_database.insert_entry(entry)

    # Now, get a new session to query the database and verify the entry was inserted
    with setup_duckdb_database.get_session() as session:
        inserted_entry = session.query(PromptMemoryEntry).filter_by(conversation_id="123").first()
        assert inserted_entry is not None
        assert inserted_entry.role == "user"
        assert inserted_entry.original_prompt_text == "Hello"
        assert inserted_entry.original_prompt_data_sha256 == "abc"


def test_insert_entry_violates_constraint(setup_duckdb_database):
    # Generate a fixed UUID
    fixed_uuid = uuid.uuid4()
    # Create two entries with the same UUID
    entry1 = PromptMemoryEntry(id=fixed_uuid, conversation_id="123", role="user", original_prompt_text="Hello")
    entry2 = PromptMemoryEntry(id=fixed_uuid, conversation_id="456", role="user", original_prompt_text="Hello again")

    # Insert the first entry
    with setup_duckdb_database.get_session() as session:
        session.add(entry1)
        session.commit()

    # Attempt to insert the second entry with the same UUID
    with setup_duckdb_database.get_session() as session:
        session.add(entry2)
        with pytest.raises(SQLAlchemyError):
            session.commit()


def test_insert_entries(setup_duckdb_database):
    entries = [
        PromptMemoryEntry(conversation_id=str(i), role="user", original_prompt_text=f"Message {i}", original_prompt_data_sha256=f"hash{i}")
        for i in range(5)
    ]

    # Now, get a new session to query the database and verify the entries were inserted
    with setup_duckdb_database.get_session() as session:
        # Use the insert_entries method to insert multiple entries into the database
        setup_duckdb_database.insert_entries(entries=entries)
        inserted_entries = session.query(PromptMemoryEntry).all()
        assert len(inserted_entries) == 5
        for i, entry in enumerate(inserted_entries):
            assert entry.conversation_id == str(i)
            assert entry.role == "user"
            assert entry.original_prompt_text == f"Message {i}"
            assert entry.original_prompt_data_sha256 == f"hash{i}"


def test_insert_embedding_entry(setup_duckdb_database):
    # Create a ConversationData entry
    conversation_entry = PromptMemoryEntry(conversation_id="123", role="user", original_prompt_text="Hello", original_prompt_data_sha256="abc")

    # Insert the ConversationData entry using the insert_entry method
    setup_duckdb_database.insert_entry(conversation_entry)

    # Re-query the ConversationData entry within a new session to ensure it's attached
    with setup_duckdb_database.get_session() as session:
        # Assuming uuid is the primary key and is set upon insertion
        reattached_conversation_entry = session.query(PromptMemoryEntry).filter_by(conversation_id="123").one()
        uuid = reattached_conversation_entry.id

    # Now that we have the uuid, we can create and insert the EmbeddingData entry
    embedding_entry = EmbeddingData(id=uuid, embedding=[1, 2, 3], embedding_type_name="test_type")
    setup_duckdb_database.insert_entry(embedding_entry)

    # Verify the EmbeddingData entry was inserted correctly
    with setup_duckdb_database.get_session() as session:
        persisted_embedding_entry = session.query(EmbeddingData).filter_by(id=uuid).first()
        assert persisted_embedding_entry is not None
        assert persisted_embedding_entry.embedding == [1, 2, 3]
        assert persisted_embedding_entry.embedding_type_name == "test_type"


def test_query_entries(setup_duckdb_database):
    # Insert some test data
    entries = [PromptMemoryEntry(conversation_id=str(i), role="user", original_prompt_text=f"Message {i}") for i in range(3)]
    setup_duckdb_database.insert_entries(entries=entries)

    # Query entries without conditions
    queried_entries = setup_duckdb_database.query_entries(PromptMemoryEntry)
    assert len(queried_entries) == 3

    # Query entries with a condition
    specific_entry = setup_duckdb_database.query_entries(
        PromptMemoryEntry, conditions=PromptMemoryEntry.conversation_id == "1"
    )
    assert len(specific_entry) == 1
    assert specific_entry[0].original_prompt_text == "Message 1"


def test_update_entries(setup_duckdb_database):
    # Insert a test entry
    entry = PromptMemoryEntry(conversation_id="123", role="user", original_prompt_text="Hello")
    setup_duckdb_database.insert_entry(entry)

    # Fetch the entry to update and update its content
    entries_to_update = setup_duckdb_database.query_entries(
        PromptMemoryEntry, conditions=PromptMemoryEntry.conversation_id == "123"
    )
    setup_duckdb_database.update_entries(entries=entries_to_update, update_fields={"original_prompt_text": "Updated Hello"})

    # Verify the entry was updated
    with setup_duckdb_database.get_session() as session:
        updated_entry = session.query(PromptMemoryEntry).filter_by(conversation_id="123").first()
        assert updated_entry.original_prompt_text == "Updated Hello"


def test_get_all_memory(setup_duckdb_database):
    # Insert some test data
    entries = [PromptMemoryEntry(conversation_id=str(i), role="user", original_prompt_text=f"Message {i}") for i in range(3)]
    setup_duckdb_database.insert_entries(entries=entries)

    # Fetch all entries
    all_entries = setup_duckdb_database.get_all_memory(PromptMemoryEntry)
    assert len(all_entries) == 3


def test_get_memories_with_json_properties(setup_duckdb_database):
    # Define a specific conversation_id
    specific_conversation_id = "test_conversation_id"

    converters = PromptConverterList([Base64Converter()]).to_json()
    target = TextTarget().to_dict()

    # Start a session
    with setup_duckdb_database.get_session() as session:
        # Create a ConversationData entry with all attributes filled
        entry = PromptMemoryEntry(
            conversation_id=specific_conversation_id,
            role="user",
            sequence=1,

            original_prompt_text="Test content",
            timestamp=datetime.datetime.utcnow(),
            original_prompt_data_sha256="test_sha256",
            labels={"normalizer_id": "id1"},
            converters=converters,
            prompt_target=target,
        )

        # Insert the ConversationData entry
        session.add(entry)
        session.commit()

        # Use the get_memories_with_conversation_id method to retrieve entries with the specific conversation_id
        retrieved_entries = setup_duckdb_database.get_memories_with_conversation_id(
            conversation_id=specific_conversation_id
        )

        # Verify that the retrieved entry matches the inserted entry
        assert len(retrieved_entries) == 1
        retrieved_entry = retrieved_entries[0]
        assert retrieved_entry.conversation_id == specific_conversation_id
        assert retrieved_entry.role == "user"
        assert retrieved_entry.original_prompt_text == "Test content"
        # For timestamp, you might want to check if it's close to the current time instead of an exact match
        assert abs((retrieved_entry.timestamp - entry.timestamp).total_seconds()) < 10  # Assuming the test runs quickly
        assert retrieved_entry.original_prompt_data_sha256 == "test_sha256"

        converters = json.loads(retrieved_entry.converters)
        assert len(converters) == 1
        assert converters[0]["__type__"] == "Base64Converter"

        prompt_target = json.loads(retrieved_entry.prompt_target)
        assert prompt_target["__type__"] == "TextTarget"

        labels = retrieved_entry.labels
        assert labels["normalizer_id"] == "id1"


def test_get_memories_with_normalizer_id(setup_duckdb_database):
    # Define a specific normalizer_id
    specific_normalizer_id = "normalizer_test_id"

    labels = {"normalizer_id": specific_normalizer_id}
    other_labels = {"normalizer_id": "other_normalizer_id"}

    # Create a list of ConversationData entries, some with the specific normalizer_id
    entries = [
        PromptMemoryEntry(
            conversation_id="123",
            role="user",
            original_prompt_text="Hello 1",
            labels=labels,
            timestamp=datetime.datetime.utcnow(),
        ),
        PromptMemoryEntry(
            conversation_id="456",
            role="user",
            original_prompt_text="Hello 2",
            labels=other_labels,
            timestamp=datetime.datetime.utcnow(),
        ),
        PromptMemoryEntry(
            conversation_id="789",
            role="user",
            original_prompt_text="Hello 3",
            labels=labels,
            timestamp=datetime.datetime.utcnow(),
        ),
    ]

    # Insert the ConversationData entries using the insert_entries method within a session
    with setup_duckdb_database.get_session() as session:
        setup_duckdb_database.insert_entries(entries=entries)
        session.commit()  # Ensure all entries are committed to the database

        # Use the get_memories_with_normalizer_id method to retrieve entries with the specific normalizer_id
        retrieved_entries = setup_duckdb_database.get_memories_with_normalizer_id(normalizer_id=specific_normalizer_id)

        # Verify that the retrieved entries match the expected normalizer_id
        assert len(retrieved_entries) == 2  # Two entries should have the specific normalizer_id
        for retrieved_entry in retrieved_entries:
            assert retrieved_entry.normalizer_id == specific_normalizer_id
            assert "Hello" in retrieved_entry.content  # Basic check to ensure content is as expected


def test_update_entries_by_conversation_id(setup_duckdb_database):
    # Define a specific conversation_id to update
    specific_conversation_id = "update_test_id"

    # Create a list of ConversationData entries, some with the specific conversation_id
    entries = [
        PromptMemoryEntry(
            conversation_id=specific_conversation_id,
            role="user",
            content="Original content 1",
            timestamp=datetime.datetime.utcnow(),
        ),
        PromptMemoryEntry(
            conversation_id="other_id", role="user", content="Original content 2", timestamp=datetime.datetime.utcnow()
        ),
        PromptMemoryEntry(
            conversation_id=specific_conversation_id,
            role="user",
            content="Original content 3",
            timestamp=datetime.datetime.utcnow(),
        ),
    ]

    # Insert the ConversationData entries using the insert_entries method within a session
    with setup_duckdb_database.get_session() as session:
        setup_duckdb_database.insert_entries(entries=entries)
        session.commit()  # Ensure all entries are committed to the database

        # Define the fields to update for entries with the specific conversation_id
        update_fields = {"content": "Updated content", "role": "assistant"}

        # Use the update_entries_by_conversation_id method to update the entries
        update_result = setup_duckdb_database.update_entries_by_conversation_id(
            conversation_id=specific_conversation_id, update_fields=update_fields
        )
        assert update_result is True  # Ensure the update operation was reported as successful

        # Verify that the entries with the specific conversation_id were updated
        updated_entries = setup_duckdb_database.query_entries(
            PromptMemoryEntry, conditions=PromptMemoryEntry.conversation_id == specific_conversation_id
        )
        for entry in updated_entries:
            assert entry.content == "Updated content"
            assert entry.role == "assistant"

        # Verify that the entry with a different conversation_id was not updated
        other_entry = session.query(PromptMemoryEntry).filter_by(conversation_id="other_id").first()
        assert other_entry.content == "Original content 2"  # Content should remain unchanged
        assert other_entry.role == "user"  # Role should remain unchanged
