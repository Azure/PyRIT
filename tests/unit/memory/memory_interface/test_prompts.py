# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from datetime import datetime
from typing import MutableSequence, Sequence
from uuid import uuid4

import pytest

from pyrit.memory import MemoryInterface
from pyrit.memory.memory_models import PromptMemoryEntry
from pyrit.models import PromptRequestPiece, Score, SeedPrompt
from pyrit.orchestrator import Orchestrator

from .conftest import assert_original_value_in_list


def test_get_prompt_request_pieces_uuid_and_string_ids(duckdb_instance: MemoryInterface):
    """Test that get_prompt_request_pieces handles both UUID objects and string representations."""
    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()
    uuid3 = uuid.uuid4()

    pieces = [
        PromptRequestPiece(
            id=uuid1,
            role="user",
            original_value="Test prompt 1",
            converted_value="Test prompt 1",
        ),
        PromptRequestPiece(
            id=uuid2,
            role="assistant",
            original_value="Test prompt 2",
            converted_value="Test prompt 2",
        ),
        PromptRequestPiece(
            id=uuid3,
            role="user",
            original_value="Test prompt 3",
            converted_value="Test prompt 3",
        ),
    ]
    duckdb_instance.add_request_pieces_to_memory(request_pieces=pieces)

    uuid_results = duckdb_instance.get_prompt_request_pieces(prompt_ids=[uuid1, uuid2])
    assert len(uuid_results) == 2
    assert {str(uuid1), str(uuid2)} == {str(piece.id) for piece in uuid_results}

    str_results = duckdb_instance.get_prompt_request_pieces(prompt_ids=[str(uuid1), str(uuid2)])
    assert len(str_results) == 2
    assert {str(uuid1), str(uuid2)} == {str(piece.id) for piece in str_results}

    mixed_types: Sequence[str | uuid.UUID] = [uuid1, str(uuid2)]
    mixed_results = duckdb_instance.get_prompt_request_pieces(prompt_ids=mixed_types)
    assert len(mixed_results) == 2
    assert {str(uuid1), str(uuid2)} == {str(piece.id) for piece in mixed_results}

    single_uuid_result = duckdb_instance.get_prompt_request_pieces(prompt_ids=[uuid3])
    assert len(single_uuid_result) == 1
    assert str(single_uuid_result[0].id) == str(uuid3)

    single_str_result = duckdb_instance.get_prompt_request_pieces(prompt_ids=[str(uuid3)])
    assert len(single_str_result) == 1
    assert str(single_str_result[0].id) == str(uuid3)


def test_get_prompt_request_pieces_labels(duckdb_instance: MemoryInterface):
    labels = {"op_name": "op1", "user_name": "name1", "harm_category": "dummy1"}
    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 1",
                labels=labels,
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="assistant",
                original_value="Hello 2",
                labels=labels,
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 3",
            )
        ),
    ]

    duckdb_instance._insert_entries(entries=entries)

    retrieved_entries = duckdb_instance.get_prompt_request_pieces(labels=labels)

    assert len(retrieved_entries) == 2  # Two entries should have the specific memory labels
    for retrieved_entry in retrieved_entries:
        assert "op_name" in retrieved_entry.labels
        assert "user_name" in retrieved_entry.labels
        assert "harm_category" in retrieved_entry.labels


def test_get_prompt_request_pieces_metadata(duckdb_instance: MemoryInterface):
    metadata: dict[str, str | int] = {"key1": "value1", "key2": "value2"}
    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 1",
                prompt_metadata=metadata,
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="assistant",
                original_value="Hello 2",
                prompt_metadata={"key2": "value2", "key3": "value3"},
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 3",
            )
        ),
    ]

    duckdb_instance._insert_entries(entries=entries)

    retrieved_entries = duckdb_instance.get_prompt_request_pieces(prompt_metadata={"key2": "value2"})

    assert len(retrieved_entries) == 2  # Two entries should have the specific memory labels
    for retrieved_entry in retrieved_entries:
        assert "key2" in retrieved_entry.prompt_metadata


def test_get_prompt_request_pieces_id(duckdb_instance: MemoryInterface):
    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 1",
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="assistant",
                original_value="Hello 2",
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 3",
            )
        ),
    ]

    id_1 = uuid.uuid4()
    id_2 = uuid.uuid4()
    entries[0].id = id_1
    entries[1].id = id_2

    duckdb_instance._insert_entries(entries=entries)

    retrieved_entries = duckdb_instance.get_prompt_request_pieces(prompt_ids=[id_1, id_2])

    assert len(retrieved_entries) == 2
    assert_original_value_in_list("Hello 1", retrieved_entries)
    assert_original_value_in_list("Hello 2", retrieved_entries)


def test_get_prompt_request_pieces_orchestrator(duckdb_instance: MemoryInterface):

    orchestrator1 = Orchestrator()
    orchestrator2 = Orchestrator()

    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 1",
                orchestrator_identifier=orchestrator1.get_identifier(),
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="assistant",
                original_value="Hello 2",
                orchestrator_identifier=orchestrator2.get_identifier(),
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 3",
                orchestrator_identifier=orchestrator1.get_identifier(),
            )
        ),
    ]

    duckdb_instance._insert_entries(entries=entries)

    orchestrator1_entries = duckdb_instance.get_prompt_request_pieces(
        orchestrator_id=orchestrator1.get_identifier()["id"]
    )

    assert len(orchestrator1_entries) == 2
    assert_original_value_in_list("Hello 1", orchestrator1_entries)
    assert_original_value_in_list("Hello 3", orchestrator1_entries)


def test_get_prompt_request_pieces_sent_after(duckdb_instance: MemoryInterface):
    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 1",
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="assistant",
                original_value="Hello 2",
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 3",
            )
        ),
    ]

    entries[0].timestamp = datetime(2022, 12, 25, 15, 30, 0)
    entries[1].timestamp = datetime(2022, 12, 25, 15, 30, 0)

    duckdb_instance._insert_entries(entries=entries)

    retrieved_entries = duckdb_instance.get_prompt_request_pieces(sent_after=datetime(2024, 1, 1))

    assert len(retrieved_entries) == 1
    assert "Hello 3" in retrieved_entries[0].original_value


def test_get_prompt_request_pieces_sent_before(duckdb_instance: MemoryInterface):
    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 1",
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="assistant",
                original_value="Hello 2",
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 3",
            )
        ),
    ]

    entries[0].timestamp = datetime(2022, 12, 25, 15, 30, 0)
    entries[1].timestamp = datetime(2021, 12, 25, 15, 30, 0)

    duckdb_instance._insert_entries(entries=entries)

    retrieved_entries = duckdb_instance.get_prompt_request_pieces(sent_before=datetime(2024, 1, 1))

    assert len(retrieved_entries) == 2
    assert_original_value_in_list("Hello 1", retrieved_entries)
    assert_original_value_in_list("Hello 2", retrieved_entries)


def test_get_prompt_request_pieces_by_value(duckdb_instance: MemoryInterface):
    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 1",
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="assistant",
                original_value="Hello 2",
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 3",
            )
        ),
    ]

    duckdb_instance._insert_entries(entries=entries)
    retrieved_entries = duckdb_instance.get_prompt_request_pieces(converted_values=["Hello 2", "Hello 3"])

    assert len(retrieved_entries) == 2
    assert_original_value_in_list("Hello 2", retrieved_entries)
    assert_original_value_in_list("Hello 3", retrieved_entries)


def test_get_prompt_request_pieces_by_hash(duckdb_instance: MemoryInterface):
    entries = [
        PromptRequestPiece(
            role="user",
            original_value="Hello 1",
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="Hello 2",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Hello 3",
        ),
    ]

    entries[0].converted_value_sha256 = "hash1"
    entries[1].converted_value_sha256 = "hash1"

    duckdb_instance.add_request_pieces_to_memory(request_pieces=entries)
    retrieved_entries = duckdb_instance.get_prompt_request_pieces(converted_value_sha256=["hash1"])

    assert len(retrieved_entries) == 2
    assert_original_value_in_list("Hello 1", retrieved_entries)
    assert_original_value_in_list("Hello 2", retrieved_entries)


@pytest.mark.asyncio
async def test_get_seed_prompts_by_hash(duckdb_instance: MemoryInterface):
    entries = [
        SeedPrompt(value="Hello 1", data_type="text"),
        SeedPrompt(value="Hello 2", data_type="text"),
    ]

    hello_1_hash = "724c531a3bc130eb46fbc4600064779552682ef4f351976fe75d876d94e8088c"

    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=entries, added_by="rlundeen")
    retrieved_entries = duckdb_instance.get_seed_prompts(value_sha256=[hello_1_hash])

    assert len(retrieved_entries) == 1
    assert retrieved_entries[0].value == "Hello 1"
    assert retrieved_entries[0].value_sha256 == hello_1_hash


def test_get_prompt_request_pieces_with_non_matching_memory_labels(duckdb_instance: MemoryInterface):
    orchestrator1 = Orchestrator()
    labels = {"op_name": "op1", "user_name": "name1", "harm_category": "dummy1"}
    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                conversation_id="123",
                role="user",
                original_value="Hello 1",
                labels=labels,
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                conversation_id="456",
                role="assistant",
                original_value="Hello 2",
                labels=labels,
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                conversation_id="789",
                role="user",
                original_value="Hello 3",
                converted_value="Hello 1",
                orchestrator_identifier=orchestrator1.get_identifier(),
            )
        ),
    ]

    duckdb_instance._insert_entries(entries=entries)
    labels = {"nonexistent_key": "nonexiststent_value"}
    retrieved_entries = duckdb_instance.get_prompt_request_pieces(labels=labels)

    assert len(retrieved_entries) == 0  # zero entries found since invalid memory labels passed


def test_get_prompt_request_pieces_sorts(
    duckdb_instance: MemoryInterface, sample_conversations: MutableSequence[PromptRequestPiece]
):
    conversation_id = sample_conversations[0].conversation_id

    # This new conversation piece should be grouped with other messages in the conversation
    sample_conversations.append(
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            conversation_id=conversation_id,
        )
    )

    duckdb_instance.add_request_pieces_to_memory(request_pieces=sample_conversations)

    response = duckdb_instance.get_prompt_request_pieces()

    current_value = response[0].conversation_id
    for obj in response[1:]:
        new_value = obj.conversation_id
        if new_value != current_value:
            if any(o.conversation_id == current_value for o in response[response.index(obj) :]):
                assert False, "Conversation IDs are not grouped together"


def test_prompt_piece_scores_duplicate_piece(duckdb_instance: MemoryInterface):
    original_id = uuid4()
    duplicate_id = uuid4()

    pieces = [
        PromptRequestPiece(
            id=original_id,
            role="assistant",
            original_value="prompt text",
        ),
        PromptRequestPiece(
            id=duplicate_id,
            role="assistant",
            original_value="prompt text",
            original_prompt_id=original_id,
        ),
    ]

    duckdb_instance.add_request_pieces_to_memory(request_pieces=pieces)

    score = Score(
        score_value=str(0.8),
        score_value_description="Sample description",
        score_type="float_scale",
        score_category="Sample category",
        score_rationale="Sample rationale",
        score_metadata="Sample metadata",
        prompt_request_response_id=original_id,
    )
    duckdb_instance.add_scores_to_memory(scores=[score])

    retrieved_pieces = duckdb_instance.get_prompt_request_pieces()

    assert len(retrieved_pieces[0].scores) == 1
    assert retrieved_pieces[0].scores[0].score_value == "0.8"

    # Check that the duplicate piece has the same score as the original
    assert len(retrieved_pieces[1].scores) == 1
    assert retrieved_pieces[1].scores[0].score_value == "0.8"


@pytest.mark.asyncio
async def test_prompt_piece_hash_stored_and_retrieved(duckdb_instance: MemoryInterface):
    entries = [
        PromptRequestPiece(
            role="user",
            original_value="Hello 1",
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="Hello 2",
        ),
    ]

    for entry in entries:
        await entry.set_sha256_values_async()

    duckdb_instance.add_request_pieces_to_memory(request_pieces=entries)
    retrieved_entries = duckdb_instance.get_prompt_request_pieces()

    assert len(retrieved_entries) == 2
    for prompt in retrieved_entries:
        assert prompt.converted_value_sha256
        assert prompt.original_value_sha256
