# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import uuid
import datetime
from unittest.mock import MagicMock

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql.sqltypes import NullType
from sqlalchemy.types import String, DateTime

from pyrit.memory.memory_models import ConversationData, EmbeddingData
from pyrit.memory import DuckDBMemory


@pytest.fixture
def setup_duckdb_database():
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
    columns = inspector.get_columns("ConversationStore")
    column_names = [col["name"] for col in columns]

    # Expected columns in ConversationData
    expected_columns = ["uuid", "role", "content", "conversation_id", "timestamp", "normalizer_id", "sha256", "labels"]
    for column in expected_columns:
        assert column in column_names, f"{column} not found in ConversationStore schema."


def test_embedding_data_schema(setup_duckdb_database):
    inspector = inspect(setup_duckdb_database.engine)
    columns = inspector.get_columns("EmbeddingStore")
    column_names = [col["name"] for col in columns]

    # Expected columns in EmbeddingData
    expected_columns = ["uuid", "embedding", "embedding_type_name"]
    for column in expected_columns:
        assert column in column_names, f"{column} not found in EmbeddingStore schema."


def test_conversation_data_column_types(setup_duckdb_database):
    inspector = inspect(setup_duckdb_database.engine)
    columns = inspector.get_columns("ConversationStore")
    column_types = {col["name"]: type(col["type"]) for col in columns}

    # Expected column types in ConversationData
    expected_column_types = {
        "uuid": UUID,
        "role": String,
        "content": String,
        "conversation_id": String,
        "timestamp": DateTime,
        "normalizer_id": String,
        "sha256": String,
        "labels": ARRAY,
    }

    for column, expected_type in expected_column_types.items():
        if column != "labels":
            assert column in column_types, f"{column} not found in ConversationStore schema."
            assert issubclass(
                column_types[column], expected_type
            ), f"Expected {column} to be a subclass of {expected_type}, got {column_types[column]} instead."

    # Handle 'labels' column separately
    assert "labels" in column_types, "'labels' column not found in ConversationStore schema."
    # Check if 'labels' column type is either NullType (due to reflection issue) or ARRAY
    assert column_types["labels"] in [NullType, ARRAY], f"Unexpected type for 'labels' column: {column_types['labels']}"


def test_embedding_data_column_types(setup_duckdb_database):
    inspector = inspect(setup_duckdb_database.engine)
    columns = inspector.get_columns("EmbeddingStore")
    column_types = {col["name"]: col["type"].__class__ for col in columns}

    # Expected column types in EmbeddingData
    expected_column_types = {
        "uuid": UUID,
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
    assert "embedding" in column_types, "'embedding' column not found in EmbeddingStore schema."
    # Check if 'embedding' column type is either NullType (due to reflection issue) or ARRAY
    assert column_types["embedding"] in [
        NullType,
        ARRAY,
    ], f"Unexpected type for 'embedding' column: {column_types['embedding']}"


def test_insert_entry(setup_duckdb_database):
    session = setup_duckdb_database.get_session()
    entry = ConversationData(
        conversation_id="123",
        role="user",
        content="Hello",
        sha256="abc",
    )
    # Use the insert_entry method to insert the entry into the database
    setup_duckdb_database.insert_entry(entry)

    # Now, get a new session to query the database and verify the entry was inserted
    with setup_duckdb_database.get_session() as session:
        inserted_entry = session.query(ConversationData).filter_by(conversation_id="123").first()
        assert inserted_entry is not None
        assert inserted_entry.role == "user"
        assert inserted_entry.content == "Hello"
        assert inserted_entry.sha256 == "abc"


def test_insert_entry_violates_constraint(setup_duckdb_database):
    # Generate a fixed UUID
    fixed_uuid = uuid.uuid4()
    # Create two entries with the same UUID
    entry1 = ConversationData(uuid=fixed_uuid, conversation_id="123", role="user", content="Hello")
    entry2 = ConversationData(uuid=fixed_uuid, conversation_id="456", role="user", content="Hello again")

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
        ConversationData(conversation_id=str(i), role="user", content=f"Message {i}", sha256=f"hash{i}")
        for i in range(5)
    ]

    # Now, get a new session to query the database and verify the entries were inserted
    with setup_duckdb_database.get_session() as session:
        # Use the insert_entries method to insert multiple entries into the database
        setup_duckdb_database.insert_entries(entries=entries)
        inserted_entries = session.query(ConversationData).all()
        assert len(inserted_entries) == 5
        for i, entry in enumerate(inserted_entries):
            assert entry.conversation_id == str(i)
            assert entry.role == "user"
            assert entry.content == f"Message {i}"
            assert entry.sha256 == f"hash{i}"


def test_insert_embedding_entry(setup_duckdb_database):
    # Create a ConversationData entry
    conversation_entry = ConversationData(conversation_id="123", role="user", content="Hello", sha256="abc")

    # Insert the ConversationData entry using the insert_entry method
    setup_duckdb_database.insert_entry(conversation_entry)

    # Re-query the ConversationData entry within a new session to ensure it's attached
    with setup_duckdb_database.get_session() as session:
        # Assuming uuid is the primary key and is set upon insertion
        reattached_conversation_entry = session.query(ConversationData).filter_by(conversation_id="123").one()
        uuid = reattached_conversation_entry.uuid

    # Now that we have the uuid, we can create and insert the EmbeddingData entry
    embedding_entry = EmbeddingData(uuid=uuid, embedding=[1, 2, 3], embedding_type_name="test_type")
    setup_duckdb_database.insert_entry(embedding_entry)

    # Verify the EmbeddingData entry was inserted correctly
    with setup_duckdb_database.get_session() as session:
        persisted_embedding_entry = session.query(EmbeddingData).filter_by(uuid=uuid).first()
        assert persisted_embedding_entry is not None
        assert persisted_embedding_entry.embedding == [1, 2, 3]
        assert persisted_embedding_entry.embedding_type_name == "test_type"


def test_query_entries(setup_duckdb_database):
    # Insert some test data
    entries = [ConversationData(conversation_id=str(i), role="user", content=f"Message {i}") for i in range(3)]
    setup_duckdb_database.insert_entries(entries=entries)

    # Query entries without conditions
    queried_entries = setup_duckdb_database.query_entries(ConversationData)
    assert len(queried_entries) == 3

    # Query entries with a condition
    specific_entry = setup_duckdb_database.query_entries(
        ConversationData, conditions=ConversationData.conversation_id == "1"
    )
    assert len(specific_entry) == 1
    assert specific_entry[0].content == "Message 1"


def test_update_entries(setup_duckdb_database):
    # Insert a test entry
    entry = ConversationData(conversation_id="123", role="user", content="Hello")
    setup_duckdb_database.insert_entry(entry)

    # Fetch the entry to update and update its content
    entries_to_update = setup_duckdb_database.query_entries(
        ConversationData, conditions=ConversationData.conversation_id == "123"
    )
    setup_duckdb_database.update_entries(entries=entries_to_update, update_fields={"content": "Updated Hello"})

    # Verify the entry was updated
    with setup_duckdb_database.get_session() as session:
        updated_entry = session.query(ConversationData).filter_by(conversation_id="123").first()
        assert updated_entry.content == "Updated Hello"


def test_get_all_memory(setup_duckdb_database):
    # Insert some test data
    entries = [ConversationData(conversation_id=str(i), role="user", content=f"Message {i}") for i in range(3)]
    setup_duckdb_database.insert_entries(entries=entries)

    # Fetch all entries
    all_entries = setup_duckdb_database.get_all_memory(ConversationData)
    assert len(all_entries) == 3


def test_get_memories_with_conversation_id(setup_duckdb_database):
    # Define a specific conversation_id
    specific_conversation_id = "test_conversation_id"

    # Start a session
    with setup_duckdb_database.get_session() as session:
        # Create a ConversationData entry with all attributes filled
        entry = ConversationData(
            conversation_id=specific_conversation_id,
            role="user",
            content="Test content",
            timestamp=datetime.datetime.utcnow(),
            normalizer_id="test_normalizer_id",
            sha256="test_sha256",
            labels=["label1", "label2"],
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
        assert retrieved_entry.content == "Test content"
        # For timestamp, you might want to check if it's close to the current time instead of an exact match
        assert abs((retrieved_entry.timestamp - entry.timestamp).total_seconds()) < 10  # Assuming the test runs quickly
        assert retrieved_entry.normalizer_id == "test_normalizer_id"
        assert retrieved_entry.sha256 == "test_sha256"
        assert retrieved_entry.labels == ["label1", "label2"]


def test_get_memories_with_normalizer_id(setup_duckdb_database):
    # Define a specific normalizer_id
    specific_normalizer_id = "normalizer_test_id"

    # Create a list of ConversationData entries, some with the specific normalizer_id
    entries = [
        ConversationData(
            conversation_id="123",
            role="user",
            content="Hello 1",
            normalizer_id=specific_normalizer_id,
            timestamp=datetime.datetime.utcnow(),
        ),
        ConversationData(
            conversation_id="456",
            role="user",
            content="Hello 2",
            normalizer_id="other_normalizer_id",
            timestamp=datetime.datetime.utcnow(),
        ),
        ConversationData(
            conversation_id="789",
            role="user",
            content="Hello 3",
            normalizer_id=specific_normalizer_id,
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
        ConversationData(
            conversation_id=specific_conversation_id,
            role="user",
            content="Original content 1",
            timestamp=datetime.datetime.utcnow(),
        ),
        ConversationData(
            conversation_id="other_id", role="user", content="Original content 2", timestamp=datetime.datetime.utcnow()
        ),
        ConversationData(
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
            ConversationData, conditions=ConversationData.conversation_id == specific_conversation_id
        )
        for entry in updated_entries:
            assert entry.content == "Updated content"
            assert entry.role == "assistant"

        # Verify that the entry with a different conversation_id was not updated
        other_entry = session.query(ConversationData).filter_by(conversation_id="other_id").first()
        assert other_entry.content == "Original content 2"  # Content should remain unchanged
        assert other_entry.role == "user"  # Role should remain unchanged
