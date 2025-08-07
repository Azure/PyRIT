# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import json
from typing import Sequence

import pytest
from sqlalchemy.inspection import inspect
from unit.mocks import get_sample_conversation_entries, get_sample_conversations

from pyrit.memory.memory_exporter import MemoryExporter
from pyrit.memory.memory_models import PromptMemoryEntry


@pytest.fixture
def sample_conversation_entries() -> Sequence[PromptMemoryEntry]:
    return get_sample_conversation_entries()


def model_to_dict(instance):
    """Converts a SQLAlchemy model instance into a dictionary."""
    return {c.key: getattr(instance, c.key) for c in inspect(instance).mapper.column_attrs}


def read_file(file_path, export_type):
    if export_type == "json":
        with open(file_path, "r") as f:
            return json.load(f)
    elif export_type == "csv":
        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return [row for row in reader]
    else:
        raise ValueError(f"Invalid export type: {export_type}")


def export(export_type, exporter, data, file_path):
    if export_type == "json":
        exporter.export_to_json(data, file_path)
    elif export_type == "csv":
        exporter.export_to_csv(data, file_path)
    else:
        raise ValueError(f"Invalid export type: {export_type}")


@pytest.mark.parametrize("export_type", ["json", "csv"])
def test_export_to_json_creates_file(tmp_path, export_type):
    exporter = MemoryExporter()
    file_path = tmp_path / f"conversations.{export_type}"
    sample_conversation_entries = get_sample_conversations()
    export(export_type=export_type, exporter=exporter, data=sample_conversation_entries, file_path=file_path)

    assert file_path.exists()  # Check that the file was created
    content = read_file(file_path=file_path, export_type=export_type)
    # Perform more detailed checks on content if necessary
    assert len(content) == 3  # Simple check for the number of items
    # Convert each PromptRequestPiece instance to a dictionary
    expected_content = [prompt_request_piece.to_dict() for prompt_request_piece in sample_conversation_entries]

    for expected, actual in zip(expected_content, content):
        assert expected["role"] == actual["role"]
        assert expected["converted_value"] == actual["converted_value"]
        assert expected["conversation_id"] == actual["conversation_id"]
        assert expected["original_value_data_type"] == actual["original_value_data_type"]
        assert expected["original_value"] == actual["original_value"]


@pytest.mark.parametrize("export_type", ["json", "csv"])
def test_export_to_json_data_with_conversations(tmp_path, export_type):
    exporter = MemoryExporter()
    sample_conversation_entries = get_sample_conversations()
    conversation_id = sample_conversation_entries[0].conversation_id

    # Define the file path using tmp_path
    file_path = tmp_path / "exported_conversations.json"

    # Call the method under test
    export(export_type=export_type, exporter=exporter, data=sample_conversation_entries, file_path=file_path)

    # Verify the file was created
    assert file_path.exists()

    # Read the file and verify its contents
    content = read_file(file_path=file_path, export_type=export_type)
    assert len(content) == 3  # Check for the expected number of items
    assert content[0]["role"] == "user"
    assert content[0]["converted_value"] == "Hello, how are you?"
    assert content[0]["conversation_id"] == conversation_id
    assert content[1]["role"] == "assistant"
    assert content[1]["converted_value"] == "I'm fine, thank you!"
    assert content[1]["conversation_id"] == conversation_id
