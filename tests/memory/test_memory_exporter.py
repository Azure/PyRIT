# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import pytest

from pyrit.memory.memory_exporter import MemoryExporter
from pyrit.memory.memory_models import ConversationData
from sqlalchemy.inspection import inspect


@pytest.fixture
def sample_conversations():
    # Create some instances of ConversationStore with sample data
    return [
        ConversationData(role="User", content="Hello, how are you?", conversation_id="12345"),
        ConversationData(role="Bot", content="I'm fine, thank you!", conversation_id="12345"),
    ]


def model_to_dict(instance):
    """Converts a SQLAlchemy model instance into a dictionary."""
    return {c.key: getattr(instance, c.key) for c in inspect(instance).mapper.column_attrs}


def test_export_to_json_creates_file(tmp_path, sample_conversations):
    exporter = MemoryExporter()
    file_path = tmp_path / "conversations.json"

    exporter.export_to_json(sample_conversations, file_path)

    assert file_path.exists()  # Check that the file was created
    with open(file_path, "r") as f:
        content = json.load(f)
        # Perform more detailed checks on content if necessary
        assert len(content) == 2  # Simple check for the number of items
        # Convert each ConversationStore instance to a dictionary
        expected_content = [model_to_dict(conv) for conv in sample_conversations]

        for expected, actual in zip(expected_content, content):
            assert expected["role"] == actual["role"]
            assert expected["content"] == actual["content"]
            assert expected["conversation_id"] == actual["conversation_id"]


def test_export_data_with_conversations(tmp_path, sample_conversations):
    exporter = MemoryExporter()

    # Define the file path using tmp_path
    file_path = tmp_path / "exported_conversations.json"

    # Call the method under test
    exporter.export_data(sample_conversations, file_path=file_path, export_type="json")

    # Verify the file was created
    assert file_path.exists()

    # Read the file and verify its contents
    with open(file_path, "r") as f:
        content = json.load(f)
        assert len(content) == 2  # Check for the expected number of items
        assert content[0]["role"] == "User"
        assert content[0]["content"] == "Hello, how are you?"
        assert content[0]["conversation_id"] == "12345"
        assert content[1]["role"] == "Bot"
        assert content[1]["content"] == "I'm fine, thank you!"
        assert content[1]["conversation_id"] == "12345"
