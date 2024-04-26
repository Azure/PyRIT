# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pytest

from pyrit.memory.memory_exporter import MemoryExporter
from pyrit.memory.memory_models import PromptMemoryEntry

from sqlalchemy.inspection import inspect
from tests.mocks import get_sample_conversation_entries


@pytest.fixture
def sample_conversation_entries() -> list[PromptMemoryEntry]:
    return get_sample_conversation_entries()


def model_to_dict(instance):
    """Converts a SQLAlchemy model instance into a dictionary."""
    return {c.key: getattr(instance, c.key) for c in inspect(instance).mapper.column_attrs}


def test_export_to_json_creates_file(tmp_path, sample_conversation_entries):
    exporter = MemoryExporter()
    file_path = tmp_path / "conversations.json"

    exporter.export_to_json(sample_conversation_entries, file_path)

    assert file_path.exists()  # Check that the file was created
    with open(file_path, "r") as f:
        content = json.load(f)
        # Perform more detailed checks on content if necessary
        assert len(content) == 3  # Simple check for the number of items
        # Convert each ConversationStore instance to a dictionary
        expected_content = [model_to_dict(conv) for conv in sample_conversation_entries]

        for expected, actual in zip(expected_content, content):
            assert expected["role"] == actual["role"]
            assert expected["converted_prompt_text"] == actual["converted_prompt_text"]
            assert expected["conversation_id"] == actual["conversation_id"]
            assert expected["original_prompt_data_type"] == actual["original_prompt_data_type"]
            assert expected["original_prompt_text"] == actual["original_prompt_text"]


def test_export_data_with_conversations(tmp_path, sample_conversation_entries):
    exporter = MemoryExporter()

    # Define the file path using tmp_path
    file_path = tmp_path / "exported_conversations.json"

    # Call the method under test
    exporter.export_data(sample_conversation_entries, file_path=file_path, export_type="json")

    # Verify the file was created
    assert file_path.exists()

    # Read the file and verify its contents
    with open(file_path, "r") as f:
        content = json.load(f)
        assert len(content) == 3  # Check for the expected number of items
        assert content[0]["role"] == "user"
        assert content[0]["converted_prompt_text"] == "Hello, how are you?"
        assert content[0]["conversation_id"] == "12345"
        assert content[1]["role"] == "assistant"
        assert content[1]["converted_prompt_text"] == "I'm fine, thank you!"
        assert content[1]["conversation_id"] == "12346"
