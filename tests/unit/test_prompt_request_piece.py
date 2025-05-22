# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from typing import MutableSequence
from unittest.mock import MagicMock

import pytest
from unit.mocks import MockPromptTarget, get_sample_conversations

from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    construct_response_from_request,
    group_conversation_request_pieces_by_sequence,
)
from pyrit.models.prompt_request_piece import sort_request_pieces
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import Base64Converter


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


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


def test_prompt_targets_serialize(patch_central_database):
    target = MockPromptTarget()
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
        converted_value="Hello",
        prompt_target_identifier=target.get_identifier(),
    )
    assert patch_central_database.called
    assert entry.prompt_target_identifier["__type__"] == "MockPromptTarget"
    assert entry.prompt_target_identifier["__module__"] == "unit.mocks"


def test_orchestrators_serialize():
    orchestrator = PromptSendingOrchestrator(objective_target=MagicMock())

    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
        converted_value="Hello",
        orchestrator_identifier=orchestrator.get_identifier(),
    )

    assert entry.orchestrator_identifier["id"] is not None
    assert entry.orchestrator_identifier["__type__"] == "PromptSendingOrchestrator"
    assert entry.orchestrator_identifier["__module__"] == "pyrit.orchestrator.single_turn.prompt_sending_orchestrator"


@pytest.mark.asyncio
async def test_hashes_generated():
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello1",
        converted_value="Hello2",
    )
    await entry.set_sha256_values_async()
    assert entry.original_value_sha256 == "948edbe7ede5aa7423476ae29dcd7d61e7711a071aea0d83698377effa896525"
    assert entry.converted_value_sha256 == "be98c2510e417405647facb89399582fc499c3de4452b3014857f92e6baad9a9"


@pytest.mark.asyncio
async def test_hashes_generated_files():
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
        await entry.set_sha256_values_async()
        assert entry.original_value_sha256 == "948edbe7ede5aa7423476ae29dcd7d61e7711a071aea0d83698377effa896525"
        assert entry.converted_value_sha256 == "948edbe7ede5aa7423476ae29dcd7d61e7711a071aea0d83698377effa896525"

    os.remove(filename)


@pytest.mark.asyncio
async def test_converted_datatype_default():
    filename = ""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filename = f.name
        f.write(b"Hello1")
        f.flush()
        f.close()
        entry = PromptRequestPiece(
            role="user",
            original_value=filename,
            original_value_data_type="image_path",
        )
        assert entry.converted_value_data_type == "image_path"
        assert entry.converted_value == filename

    os.remove(filename)


def test_hashes_generated_files_unknown_type():
    with pytest.raises(ValueError, match="is not a valid data type."):
        PromptRequestPiece(
            role="user",
            original_value="Hello1",
            original_value_data_type="new_unknown_type",  # type: ignore
        )


def test_prompt_response_get_value(sample_conversations: MutableSequence[PromptRequestPiece]):
    request_response = PromptRequestResponse(request_pieces=sample_conversations)
    assert request_response.get_value() == "Hello, how are you?"
    assert request_response.get_value(1) == "I'm fine, thank you!"

    with pytest.raises(IndexError):
        request_response.get_value(3)


def test_prompt_response_get_values(sample_conversations: MutableSequence[PromptRequestPiece]):
    request_response = PromptRequestResponse(request_pieces=sample_conversations)
    assert request_response.get_values() == ["Hello, how are you?", "I'm fine, thank you!", "I'm fine, thank you!"]


def test_prompt_response_validate(sample_conversations: MutableSequence[PromptRequestPiece]):
    for c in sample_conversations:
        c.conversation_id = sample_conversations[0].conversation_id
        c.role = sample_conversations[0].role

    request_response = PromptRequestResponse(request_pieces=sample_conversations)
    request_response.validate()


def test_prompt_response_empty_throws():
    request_response = PromptRequestResponse(request_pieces=[])
    with pytest.raises(ValueError, match="Empty request pieces."):
        request_response.validate()


def test_prompt_response_validate_conversation_id_throws(sample_conversations: MutableSequence[PromptRequestPiece]):
    for c in sample_conversations:
        c.role = "user"
        c.conversation_id = str(uuid.uuid4())

    request_response = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="Conversation ID mismatch."):
        request_response.validate()


def test_prompt_request_response_inconsistent_roles_throws(sample_conversations: MutableSequence[PromptRequestPiece]):
    for c in sample_conversations:
        c.conversation_id = sample_conversations[0].conversation_id

    request_response = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="Inconsistent roles within the same prompt request response entry."):
        request_response.validate()


def test_group_conversation_request_pieces_throws(sample_conversations: MutableSequence[PromptRequestPiece]):
    with pytest.raises(ValueError, match="Conversation ID must match."):
        group_conversation_request_pieces_by_sequence(sample_conversations)


def test_group_conversation_request_pieces(sample_conversations: MutableSequence[PromptRequestPiece]):
    convo_group = [
        entry for entry in sample_conversations if entry.conversation_id == sample_conversations[0].conversation_id
    ]
    groups = group_conversation_request_pieces_by_sequence(convo_group)
    assert groups
    assert len(groups) == 1
    assert groups[0].request_pieces[0].sequence == 0


def test_group_conversation_request_pieces_multiple_groups(sample_conversations: MutableSequence[PromptRequestPiece]):
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
async def test_prompt_request_piece_sets_original_sha256():
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
    )

    entry.original_value = "newvalue"
    await entry.set_sha256_values_async()
    assert entry.original_value_sha256 == "70e01503173b8e904d53b40b3ebb3bded5e5d3add087d3463a4b1abe92f1a8ca"


@pytest.mark.asyncio
async def test_prompt_request_piece_sets_converted_sha256():
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
    )
    entry.converted_value = "newvalue"
    await entry.set_sha256_values_async()
    assert entry.converted_value_sha256 == "70e01503173b8e904d53b40b3ebb3bded5e5d3add087d3463a4b1abe92f1a8ca"


def test_order_request_pieces_by_conversation_single_conversation():
    pieces = [
        PromptRequestPiece(
            role="user",
            id="prompt-1",
            original_value="Hello 1",
            conversation_id="conv1",
            timestamp=datetime.now() - timedelta(seconds=10),
            sequence=2,
        ),
        PromptRequestPiece(
            role="user",
            id="prompt-2",
            original_value="Hello 2",
            conversation_id="conv1",
            timestamp=datetime.now() - timedelta(seconds=10),
            sequence=1,
        ),
        PromptRequestPiece(
            role="user",
            id="prompt-3",
            original_value="Hello 3",
            conversation_id="conv1",
            timestamp=datetime.now(),
            sequence=3,
        ),
    ]

    expected = [
        PromptRequestPiece(
            role="user",
            original_value="Hello 2",
            conversation_id="conv1",
            timestamp=pieces[1].timestamp,
            sequence=1,
            id="prompt-2",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 1",
            conversation_id="conv1",
            timestamp=pieces[0].timestamp,
            sequence=2,
            id="prompt-1",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 3",
            conversation_id="conv1",
            timestamp=pieces[2].timestamp,
            sequence=3,
            id="prompt-3",
        ),
    ]

    ordered = sort_request_pieces(pieces)
    assert ordered == expected


def test_order_request_pieces_by_conversation_multiple_conversations():
    pieces = [
        PromptRequestPiece(
            role="user",
            original_value="Hello 4",
            conversation_id="conv2",
            timestamp=datetime.now() - timedelta(seconds=5),
            sequence=2,
            id="4",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 1",
            conversation_id="conv1",
            timestamp=datetime.now() - timedelta(seconds=15),
            sequence=1,
            id="1",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 3",
            conversation_id="conv2",
            timestamp=datetime.now() - timedelta(seconds=10),
            sequence=1,
            id="3",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 2",
            conversation_id="conv1",
            timestamp=datetime.now() - timedelta(seconds=10),
            sequence=2,
            id="2",
        ),
    ]

    expected = [
        PromptRequestPiece(
            role="user",
            original_value="Hello 1",
            conversation_id="conv1",
            timestamp=pieces[1].timestamp,
            sequence=1,
            id="1",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 2",
            conversation_id="conv1",
            timestamp=pieces[3].timestamp,
            sequence=2,
            id="2",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 3",
            conversation_id="conv2",
            timestamp=pieces[2].timestamp,
            sequence=1,
            id="3",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 4",
            conversation_id="conv2",
            timestamp=pieces[0].timestamp,
            sequence=2,
            id="4",
        ),
    ]

    assert sort_request_pieces(pieces) == expected


def test_order_request_pieces_by_conversation_same_timestamp():
    timestamp = datetime.now()

    pieces = [
        PromptRequestPiece(
            role="user",
            original_value="Hello 4",
            conversation_id="conv2",
            timestamp=timestamp,
            sequence=2,
            id="4",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 1",
            conversation_id="conv1",
            timestamp=timestamp,
            sequence=1,
            id="1",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 3",
            conversation_id="conv2",
            timestamp=timestamp,
            sequence=1,
            id="3",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 2",
            conversation_id="conv1",
            timestamp=timestamp,
            sequence=2,
            id="2",
        ),
    ]

    expected = [
        PromptRequestPiece(
            role="user",
            original_value="Hello 1",
            conversation_id="conv1",
            timestamp=pieces[1].timestamp,
            sequence=1,
            id="1",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 2",
            conversation_id="conv1",
            timestamp=pieces[3].timestamp,
            sequence=2,
            id="2",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 3",
            conversation_id="conv2",
            timestamp=pieces[2].timestamp,
            sequence=1,
            id="3",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 4",
            conversation_id="conv2",
            timestamp=pieces[0].timestamp,
            sequence=2,
            id="4",
        ),
    ]

    sorted = sort_request_pieces(pieces)
    assert sorted == expected


def test_order_request_pieces_by_conversation_empty_list():
    pieces = []
    expected = []
    assert sort_request_pieces(pieces) == expected


def test_order_request_pieces_by_conversation_single_message():
    pieces = [PromptRequestPiece(role="user", original_value="Hello 1", conversation_id="conv1", id="1")]
    expected = [PromptRequestPiece(role="user", original_value="Hello 1", conversation_id="conv1", id="1")]

    assert sort_request_pieces(pieces) == expected


def test_order_request_pieces_by_conversation_same_timestamp_different_sequences():
    pieces = [
        PromptRequestPiece(
            role="user", original_value="Hello 2", conversation_id="conv1", timestamp=datetime.now(), sequence=2, id="2"
        ),
        PromptRequestPiece(
            role="user", original_value="Hello 1", conversation_id="conv1", timestamp=datetime.now(), sequence=1, id="1"
        ),
    ]
    for i, piece in enumerate(pieces):
        piece.prompt_id = f"prompt-{i}"
    expected = [
        PromptRequestPiece(
            role="user",
            original_value="Hello 1",
            conversation_id="conv1",
            timestamp=pieces[1].timestamp,
            sequence=1,
            id="1",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 2",
            conversation_id="conv1",
            timestamp=pieces[0].timestamp,
            sequence=2,
            id="2",
        ),
    ]

    assert sort_request_pieces(pieces) == expected


def test_prompt_request_piece_to_dict():
    entry = PromptRequestPiece(
        role="user",
        original_value="Hello",
        converted_value="Hello",
        conversation_id="test_conversation",
        sequence=1,
        labels={"label1": "value1"},
        prompt_metadata="metadata",
        converter_identifiers=[
            {"__type__": "Base64Converter", "__module__": "pyrit.prompt_converter.base64_converter"}
        ],
        prompt_target_identifier={"__type__": "MockPromptTarget", "__module__": "unit.mocks"},
        orchestrator_identifier={
            "id": str(uuid.uuid4()),
            "__type__": "PromptSendingOrchestrator",
            "__module__": "pyrit.orchestrator.single_turn.prompt_sending_orchestrator",
        },
        scorer_identifier={"key": "value"},
        original_value_data_type="text",
        converted_value_data_type="text",
        response_error="none",
        originator="undefined",
        original_prompt_id=uuid.uuid4(),
        timestamp=datetime.now(),
        scores=[
            Score(
                id=str(uuid.uuid4()),
                score_value="false",
                score_value_description="true false score",
                score_type="true_false",
                score_category="Category1",
                score_rationale="Rationale text",
                score_metadata={"key": "value"},
                scorer_class_identifier="Scorer1",
                prompt_request_response_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                task="Task1",
            )
        ],
    )

    result = entry.to_dict()

    expected_keys = [
        "id",
        "role",
        "conversation_id",
        "sequence",
        "timestamp",
        "labels",
        "prompt_metadata",
        "converter_identifiers",
        "prompt_target_identifier",
        "orchestrator_identifier",
        "scorer_identifier",
        "original_value_data_type",
        "original_value",
        "original_value_sha256",
        "converted_value_data_type",
        "converted_value",
        "converted_value_sha256",
        "response_error",
        "originator",
        "original_prompt_id",
        "scores",
    ]

    for key in expected_keys:
        assert key in result, f"Missing key: {key}"

    assert result["id"] == str(entry.id)
    assert result["role"] == entry.role
    assert result["conversation_id"] == entry.conversation_id
    assert result["sequence"] == entry.sequence
    assert result["timestamp"] == entry.timestamp.isoformat()
    assert result["labels"] == entry.labels
    assert result["prompt_metadata"] == entry.prompt_metadata
    assert result["converter_identifiers"] == entry.converter_identifiers
    assert result["prompt_target_identifier"] == entry.prompt_target_identifier
    assert result["orchestrator_identifier"] == entry.orchestrator_identifier
    assert result["scorer_identifier"] == entry.scorer_identifier
    assert result["original_value_data_type"] == entry.original_value_data_type
    assert result["original_value"] == entry.original_value
    assert result["original_value_sha256"] == entry.original_value_sha256
    assert result["converted_value_data_type"] == entry.converted_value_data_type
    assert result["converted_value"] == entry.converted_value
    assert result["converted_value_sha256"] == entry.converted_value_sha256
    assert result["response_error"] == entry.response_error
    assert result["originator"] == entry.originator
    assert result["original_prompt_id"] == str(entry.original_prompt_id)
    assert result["scores"] == [score.to_dict() for score in entry.scores]


def test_construct_response_from_request_combines_metadata():
    # Create a request piece with metadata
    request = PromptRequestPiece(
        role="user", original_value="test prompt", conversation_id="123", prompt_metadata={"key1": "value1", "key2": 2}
    )

    additional_metadata = {"key2": 3, "key3": "value3"}

    response = construct_response_from_request(
        request=request, response_text_pieces=["test response"], prompt_metadata=additional_metadata
    )

    assert len(response.request_pieces) == 1
    response_piece = response.request_pieces[0]

    assert response_piece.prompt_metadata["key1"] == "value1"  # Original value preserved
    assert response_piece.prompt_metadata["key2"] == 3  # Overridden by additional metadata
    assert response_piece.prompt_metadata["key3"] == "value3"  # Added from additional metadata

    assert response_piece.role == "assistant"
    assert response_piece.original_value == "test response"
    assert response_piece.conversation_id == "123"
    assert response_piece.original_value_data_type == "text"
    assert response_piece.converted_value_data_type == "text"
    assert response_piece.response_error == "none"


def test_construct_response_from_request_no_metadata():
    request = PromptRequestPiece(role="user", original_value="test prompt", conversation_id="123")

    response = construct_response_from_request(request=request, response_text_pieces=["test response"])

    assert len(response.request_pieces) == 1
    response_piece = response.request_pieces[0]

    assert not response_piece.prompt_metadata

    assert response_piece.role == "assistant"
    assert response_piece.original_value == "test response"
    assert response_piece.conversation_id == "123"
    assert response_piece.original_value_data_type == "text"
    assert response_piece.converted_value_data_type == "text"
    assert response_piece.response_error == "none"
