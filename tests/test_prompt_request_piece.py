# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
import uuid
import pytest
import time

from datetime import datetime
from unittest.mock import MagicMock, patch
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse, group_conversation_request_pieces_by_sequence
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import Base64Converter
from tests.mocks import MockPromptTarget
from pyrit.memory import DuckDBMemory

from tests.mocks import get_sample_conversations


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture(scope="function")
def set_duckdb_in_memory():
    duckdb_in_memory = DuckDBMemory(db_path=":memory:")
    CentralMemory.set_memory_instance(duckdb_in_memory)


@pytest.fixture
def mock_memory_instance():
    """Fixture to mock CentralMemory.get_memory_instance returning None"""
    with patch.object(CentralMemory, "get_memory_instance", return_value=None) as mock:
        yield mock


def test_id_set():
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
        converted_value="Hello",
    )
    assert entry.id is not None


def test_datetime_set():
    now = datetime.now()
    time.sleep(0.1)
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
        converted_value="Hello",
    )
    assert entry.timestamp > now


def test_converters_serialize():
    converter_identifiers = [Base64Converter().get_identifier()]
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
        converted_value="Hello",
        converter_identifiers=converter_identifiers,
    )

    assert len(entry.converter_identifiers) == 1

    converter = entry.converter_identifiers[0]

    assert converter["__type__"] == "Base64Converter"
    assert converter["__module__"] == "pyrit.prompt_converter.base64_converter"


def test_prompt_targets_serialize(mock_memory_instance):
    target = MockPromptTarget()
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
        converted_value="Hello",
        prompt_target_identifier=target.get_identifier(),
    )
    assert mock_memory_instance.called
    assert entry.prompt_target_identifier["__type__"] == "MockPromptTarget"
    assert entry.prompt_target_identifier["__module__"] == "tests.mocks"


def test_orchestrators_serialize(mock_memory_instance):
    orchestrator = PromptSendingOrchestrator(prompt_target=MagicMock())

    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
        converted_value="Hello",
        orchestrator_identifier=orchestrator.get_identifier(),
    )

    assert entry.orchestrator_identifier["id"] is not None
    assert entry.orchestrator_identifier["__type__"] == "PromptSendingOrchestrator"
    assert entry.orchestrator_identifier["__module__"] == "pyrit.orchestrator.prompt_sending_orchestrator"


@pytest.mark.asyncio
async def test_hashes_generated(set_duckdb_in_memory):
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello1",
        converted_value="Hello2",
    )
    await entry.compute_sha256()
    assert entry.original_value_sha256 == "948edbe7ede5aa7423476ae29dcd7d61e7711a071aea0d83698377effa896525"
    assert entry.converted_value_sha256 == "be98c2510e417405647facb89399582fc499c3de4452b3014857f92e6baad9a9"


@pytest.mark.asyncio
async def test_hashes_generated_files(set_duckdb_in_memory):
    filename = ""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filename = f.name
        f.write(b"Hello1")
        f.flush()
        f.close()
        entry = PromptRequestPiece(
            role="user",
            original_value=filename,
            converted_value=filename,
            original_value_data_type="image_path",
            converted_value_data_type="audio_path",
        )
        await entry.compute_sha256()
        assert entry.original_value_sha256 == "948edbe7ede5aa7423476ae29dcd7d61e7711a071aea0d83698377effa896525"
        assert entry.converted_value_sha256 == "948edbe7ede5aa7423476ae29dcd7d61e7711a071aea0d83698377effa896525"

    os.remove(filename)


def test_hashes_generated_files_unknown_type():
    with pytest.raises(ValueError, match="is not a valid data type."):
        PromptRequestPiece(
            role="user",
            original_value="Hello1",
            original_value_data_type="new_unknown_type",  # type: ignore
        )


def test_prompt_response_validate(mock_memory_instance, sample_conversations: list[PromptRequestPiece]):
    for c in sample_conversations:
        c.conversation_id = sample_conversations[0].conversation_id
        c.role = sample_conversations[0].role

    request_response = PromptRequestResponse(request_pieces=sample_conversations)
    request_response.validate()


def test_prompt_response_empty_throws():
    request_response = PromptRequestResponse(request_pieces=[])
    with pytest.raises(ValueError, match="Empty request pieces."):
        request_response.validate()


def test_prompt_response_validate_conversation_id_throws(sample_conversations: list[PromptRequestPiece]):
    for c in sample_conversations:
        c.role = "user"
        c.conversation_id = str(uuid.uuid4())

    request_response = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="Conversation ID mismatch."):
        request_response.validate()


def test_prompt_request_response_inconsistent_roles_throws(
    mock_memory_instance, sample_conversations: list[PromptRequestPiece]
):
    for c in sample_conversations:
        c.conversation_id = sample_conversations[0].conversation_id

    request_response = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="Inconsistent roles within the same prompt request response entry."):
        request_response.validate()


def test_group_conversation_request_pieces_throws(mock_memory_instance, sample_conversations: list[PromptRequestPiece]):
    with pytest.raises(ValueError, match="Conversation ID must match."):
        group_conversation_request_pieces_by_sequence(sample_conversations)


def test_group_conversation_request_pieces(mock_memory_instance, sample_conversations: list[PromptRequestPiece]):
    convo_group = [
        entry for entry in sample_conversations if entry.conversation_id == sample_conversations[0].conversation_id
    ]
    groups = group_conversation_request_pieces_by_sequence(convo_group)
    assert groups
    assert len(groups) == 1
    assert groups[0].request_pieces[0].sequence == 0


def test_group_conversation_request_pieces_multiple_groups(
    mock_memory_instance, sample_conversations: list[PromptRequestPiece]
):
    convo_group = [
        entry for entry in sample_conversations if entry.conversation_id == sample_conversations[0].conversation_id
    ]
    convo_group.append(
        PromptRequestPiece(
            role="user",
            original_value="Hello",
            conversation_id=convo_group[0].conversation_id,
            sequence=1,
        )
    )
    groups = group_conversation_request_pieces_by_sequence(convo_group)
    assert groups
    assert len(groups) == 2
    assert groups[0].request_pieces[0].sequence == 0
    assert groups[1].request_pieces[0].sequence == 1


def test_prompt_request_piece_no_roles():
    with pytest.raises(ValueError) as excinfo:
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="",  # type: ignore
                    converted_value_data_type="text",
                    original_value="Hello",
                    converted_value="Hello",
                )
            ]
        )

        assert "not a valid role." in str(excinfo.value)


@pytest.mark.asyncio
async def test_prompt_request_piece_sets_original_sha256(set_duckdb_in_memory):
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
    )

    entry.original_value = "newvalue"
    await entry.compute_sha256()
    assert entry.original_value_sha256 == "70e01503173b8e904d53b40b3ebb3bded5e5d3add087d3463a4b1abe92f1a8ca"


@pytest.mark.asyncio
async def test_prompt_request_piece_sets_converted_sha256(set_duckdb_in_memory):
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
    )
    entry.converted_value = "newvalue"
    await entry.compute_sha256()
    assert entry.converted_value_sha256 == "70e01503173b8e904d53b40b3ebb3bded5e5d3add087d3463a4b1abe92f1a8ca"
