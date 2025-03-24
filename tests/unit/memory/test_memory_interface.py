# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from string import ascii_lowercase
from typing import Literal, MutableSequence, Sequence
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from unit.mocks import get_sample_conversation_entries, get_sample_conversations

from pyrit.common.path import DB_DATA_PATH
from pyrit.memory import MemoryExporter, MemoryInterface, PromptMemoryEntry
from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.orchestrator import Orchestrator


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def sample_conversation_entries() -> Sequence[PromptMemoryEntry]:
    return get_sample_conversation_entries()


def generate_random_string(length: int = 10) -> str:
    return "".join(random.choice(ascii_lowercase) for _ in range(length))


def assert_original_value_in_list(original_value: str, prompt_request_pieces: Sequence[PromptRequestPiece]):
    for piece in prompt_request_pieces:
        if piece.original_value == original_value:
            return True
    raise AssertionError(f"Original value {original_value} not found in list")


def test_memory(duckdb_instance: MemoryInterface):
    assert duckdb_instance


def test_conversation_memory_empty_by_default(duckdb_instance: MemoryInterface):
    expected_count = 0
    c = duckdb_instance.get_prompt_request_pieces()
    assert len(c) == expected_count


@pytest.mark.parametrize("num_conversations", [1, 2, 3])
def test_add_request_pieces_to_memory(
    duckdb_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece], num_conversations: int
):
    for c in sample_conversations[:num_conversations]:
        c.conversation_id = sample_conversations[0].conversation_id
        c.role = sample_conversations[0].role

    request_response = PromptRequestResponse(request_pieces=sample_conversations[:num_conversations])

    duckdb_instance.add_request_response_to_memory(request=request_response)
    assert len(duckdb_instance.get_prompt_request_pieces()) == num_conversations


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


def test_duplicate_memory(duckdb_instance: MemoryInterface):
    orchestrator1 = Orchestrator()
    orchestrator2 = Orchestrator()
    conversation_id_1 = "11111"
    conversation_id_2 = "22222"
    conversation_id_3 = "33333"
    pieces = [
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id_1,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id_1,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id_3,
            orchestrator_identifier=orchestrator2.get_identifier(),
        ),
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id_2,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id_2,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
    ]
    duckdb_instance.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(duckdb_instance.get_prompt_request_pieces()) == 5
    orchestrator3 = Orchestrator()
    new_conversation_id1 = duckdb_instance.duplicate_conversation(
        new_orchestrator_id=orchestrator3.get_identifier()["id"],
        conversation_id=conversation_id_1,
    )
    new_conversation_id2 = duckdb_instance.duplicate_conversation(
        new_orchestrator_id=orchestrator3.get_identifier()["id"],
        conversation_id=conversation_id_2,
    )
    all_pieces = duckdb_instance.get_prompt_request_pieces()
    assert len(all_pieces) == 9
    assert len([p for p in all_pieces if p.orchestrator_identifier["id"] == orchestrator1.get_identifier()["id"]]) == 4
    assert len([p for p in all_pieces if p.orchestrator_identifier["id"] == orchestrator2.get_identifier()["id"]]) == 1
    assert len([p for p in all_pieces if p.orchestrator_identifier["id"] == orchestrator3.get_identifier()["id"]]) == 4
    assert len([p for p in all_pieces if p.conversation_id == conversation_id_1]) == 2
    assert len([p for p in all_pieces if p.conversation_id == conversation_id_2]) == 2
    assert len([p for p in all_pieces if p.conversation_id == conversation_id_3]) == 1
    assert len([p for p in all_pieces if p.conversation_id == new_conversation_id1]) == 2
    assert len([p for p in all_pieces if p.conversation_id == new_conversation_id2]) == 2


# Ensure that the score entries are not duplicated when a conversation is duplicated
def test_duplicate_conversation_pieces_not_score(duckdb_instance: MemoryInterface):
    conversation_id = str(uuid4())
    prompt_id_1 = uuid4()
    prompt_id_2 = uuid4()
    orchestrator1 = Orchestrator()
    memory_labels = {"sample": "label"}
    pieces = [
        PromptRequestPiece(
            id=prompt_id_1,
            role="assistant",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
            labels=memory_labels,
        ),
        PromptRequestPiece(
            id=prompt_id_2,
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
            labels=memory_labels,
        ),
    ]
    # Ensure that the original prompt id defaults to the id of the piece
    assert pieces[0].original_prompt_id == pieces[0].id
    assert pieces[1].original_prompt_id == pieces[1].id
    scores = [
        Score(
            score_value=str(0.8),
            score_value_description="High score",
            score_type="float_scale",
            score_category="test",
            score_rationale="Test score",
            score_metadata="Test metadata",
            scorer_class_identifier={"__type__": "TestScorer1"},
            prompt_request_response_id=prompt_id_1,
        ),
        Score(
            score_value=str(0.5),
            score_value_description="High score",
            score_type="float_scale",
            score_category="test",
            score_rationale="Test score",
            score_metadata="Test metadata",
            scorer_class_identifier={"__type__": "TestScorer2"},
            prompt_request_response_id=prompt_id_2,
        ),
    ]
    duckdb_instance.add_request_pieces_to_memory(request_pieces=pieces)
    duckdb_instance.add_scores_to_memory(scores=scores)
    orchestrator2 = Orchestrator()
    new_conversation_id = duckdb_instance.duplicate_conversation(
        new_orchestrator_id=orchestrator2.get_identifier()["id"],
        conversation_id=conversation_id,
    )
    new_pieces = duckdb_instance.get_prompt_request_pieces(conversation_id=new_conversation_id)
    new_pieces_ids = [str(p.id) for p in new_pieces]
    assert len(new_pieces) == 2
    original_ids = {piece.original_prompt_id for piece in new_pieces}
    assert original_ids == {prompt_id_1, prompt_id_2}

    for piece in new_pieces:
        assert piece.id not in (prompt_id_1, prompt_id_2)
    assert len(duckdb_instance.get_scores_by_memory_labels(memory_labels=memory_labels)) == 2
    assert len(duckdb_instance.get_scores_by_orchestrator_id(orchestrator_id=orchestrator1.get_identifier()["id"])) == 2
    assert len(duckdb_instance.get_scores_by_orchestrator_id(orchestrator_id=orchestrator2.get_identifier()["id"])) == 2
    # The duplicate prompts ids should not have scores so only two scores are returned
    assert (
        len(
            duckdb_instance.get_scores_by_prompt_ids(
                prompt_request_response_ids=[str(prompt_id_1), str(prompt_id_2)] + new_pieces_ids
            )
        )
        == 2
    )


def test_duplicate_conversation_excluding_last_turn(duckdb_instance: MemoryInterface):
    orchestrator1 = Orchestrator()
    orchestrator2 = Orchestrator()
    conversation_id_1 = "11111"
    conversation_id_2 = "22222"
    pieces = [
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=1,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            sequence=2,
            conversation_id=conversation_id_1,
            orchestrator_identifier=orchestrator2.get_identifier(),
        ),
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id_2,
            sequence=2,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id_2,
            sequence=3,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
    ]
    duckdb_instance.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(duckdb_instance.get_prompt_request_pieces()) == 5
    orchestrator3 = Orchestrator()

    new_conversation_id1 = duckdb_instance.duplicate_conversation_excluding_last_turn(
        new_orchestrator_id=orchestrator3.get_identifier()["id"],
        conversation_id=conversation_id_1,
    )

    all_memory = duckdb_instance.get_prompt_request_pieces()
    assert len(all_memory) == 7

    duplicate_conversation = duckdb_instance.get_prompt_request_pieces(conversation_id=new_conversation_id1)
    assert len(duplicate_conversation) == 2

    for piece in duplicate_conversation:
        assert piece.sequence < 2


def test_duplicate_conversation_excluding_last_turn_not_score(duckdb_instance: MemoryInterface):
    conversation_id = str(uuid4())
    prompt_id_1 = uuid4()
    prompt_id_2 = uuid4()
    orchestrator1 = Orchestrator()
    memory_labels = {"sample": "label"}
    pieces = [
        PromptRequestPiece(
            id=prompt_id_1,
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
            labels=memory_labels,
        ),
        PromptRequestPiece(
            id=prompt_id_2,
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id,
            sequence=1,
            orchestrator_identifier=orchestrator1.get_identifier(),
            labels=memory_labels,
        ),
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="That's good.",
            conversation_id=conversation_id,
            sequence=2,
            orchestrator_identifier=orchestrator1.get_identifier(),
            labels=memory_labels,
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="Thanks.",
            conversation_id=conversation_id,
            sequence=3,
            orchestrator_identifier=orchestrator1.get_identifier(),
            labels=memory_labels,
        ),
    ]
    # Ensure that the original prompt id defaults to the id of the piece
    assert pieces[0].original_prompt_id == pieces[0].id
    assert pieces[1].original_prompt_id == pieces[1].id
    scores = [
        Score(
            score_value=str(0.8),
            score_value_description="High score",
            score_type="float_scale",
            score_category="test",
            score_rationale="Test score",
            score_metadata="Test metadata",
            scorer_class_identifier={"__type__": "TestScorer1"},
            prompt_request_response_id=prompt_id_1,
        ),
        Score(
            score_value=str(0.5),
            score_value_description="High score",
            score_type="float_scale",
            score_category="test",
            score_rationale="Test score",
            score_metadata="Test metadata",
            scorer_class_identifier={"__type__": "TestScorer2"},
            prompt_request_response_id=prompt_id_2,
        ),
    ]
    duckdb_instance.add_request_pieces_to_memory(request_pieces=pieces)
    duckdb_instance.add_scores_to_memory(scores=scores)
    orchestrator2 = Orchestrator()

    new_conversation_id = duckdb_instance.duplicate_conversation_excluding_last_turn(
        new_orchestrator_id=orchestrator2.get_identifier()["id"],
        conversation_id=conversation_id,
    )
    new_pieces = duckdb_instance.get_prompt_request_pieces(conversation_id=new_conversation_id)
    new_pieces_ids = [str(p.id) for p in new_pieces]
    assert len(new_pieces) == 2
    assert new_pieces[0].original_prompt_id == prompt_id_1
    assert new_pieces[1].original_prompt_id == prompt_id_2
    assert new_pieces[0].id != prompt_id_1
    assert new_pieces[1].id != prompt_id_2
    assert len(duckdb_instance.get_scores_by_memory_labels(memory_labels=memory_labels)) == 2
    assert len(duckdb_instance.get_scores_by_orchestrator_id(orchestrator_id=orchestrator1.get_identifier()["id"])) == 2
    assert len(duckdb_instance.get_scores_by_orchestrator_id(orchestrator_id=orchestrator2.get_identifier()["id"])) == 2
    # The duplicate prompts ids should not have scores so only two scores are returned
    assert (
        len(
            duckdb_instance.get_scores_by_prompt_ids(
                prompt_request_response_ids=[str(prompt_id_1), str(prompt_id_2)] + new_pieces_ids
            )
        )
        == 2
    )


def test_duplicate_conversation_excluding_last_turn_same_orchestrator(duckdb_instance: MemoryInterface):
    orchestrator1 = Orchestrator()
    conversation_id_1 = "11111"
    pieces = [
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=1,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=2,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=3,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
    ]
    duckdb_instance.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(duckdb_instance.get_prompt_request_pieces()) == 4

    new_conversation_id1 = duckdb_instance.duplicate_conversation_excluding_last_turn(
        conversation_id=conversation_id_1,
    )

    all_memory = duckdb_instance.get_prompt_request_pieces()
    assert len(all_memory) == 6

    duplicate_conversation = duckdb_instance.get_prompt_request_pieces(conversation_id=new_conversation_id1)
    assert len(duplicate_conversation) == 2

    for piece in duplicate_conversation:
        assert piece.sequence < 2


def test_duplicate_memory_orchestrator_id_collision(duckdb_instance: MemoryInterface):
    orchestrator1 = Orchestrator()
    conversation_id = "11111"
    pieces = [
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
    ]
    duckdb_instance.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(duckdb_instance.get_prompt_request_pieces()) == 1
    with pytest.raises(ValueError):
        duckdb_instance.duplicate_conversation(
            new_orchestrator_id=str(orchestrator1.get_identifier()["id"]),
            conversation_id=conversation_id,
        )


def test_add_request_pieces_to_memory_calls_validate(duckdb_instance: MemoryInterface):
    request_response = MagicMock(PromptRequestResponse)
    request_response.request_pieces = [MagicMock(PromptRequestPiece)]
    with (
        patch("pyrit.memory.duckdb_memory.DuckDBMemory.add_request_pieces_to_memory"),
        patch("pyrit.memory.memory_interface.MemoryInterface._update_sequence"),
    ):
        duckdb_instance.add_request_response_to_memory(request=request_response)
    assert request_response.validate.called


def test_add_request_pieces_to_memory_updates_sequence(
    duckdb_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):
    for conversation in sample_conversations:
        conversation.conversation_id = sample_conversations[0].conversation_id
        conversation.role = sample_conversations[0].role
        conversation.sequence = 17

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory.add_request_pieces_to_memory") as mock_add:
        duckdb_instance.add_request_response_to_memory(
            request=PromptRequestResponse(request_pieces=sample_conversations)
        )
        assert mock_add.called

        args, kwargs = mock_add.call_args
        assert kwargs["request_pieces"][0].sequence == 0, "Sequence should be reset to 0"
        assert kwargs["request_pieces"][1].sequence == 0, "Sequence should be reset to 0"
        assert kwargs["request_pieces"][2].sequence == 0, "Sequence should be reset to 0"


def test_add_request_pieces_to_memory_updates_sequence_with_prev_conversation(
    duckdb_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):

    for conversation in sample_conversations:
        conversation.conversation_id = sample_conversations[0].conversation_id
        conversation.role = sample_conversations[0].role
        conversation.sequence = 17

    # insert one of these into memory
    duckdb_instance.add_request_response_to_memory(request=PromptRequestResponse(request_pieces=sample_conversations))

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory.add_request_pieces_to_memory") as mock_add:
        duckdb_instance.add_request_response_to_memory(
            request=PromptRequestResponse(request_pieces=sample_conversations)
        )
        assert mock_add.called

        args, kwargs = mock_add.call_args
        assert kwargs["request_pieces"][0].sequence == 1, "Sequence should increment previous conversation by 1"
        assert kwargs["request_pieces"][1].sequence == 1
        assert kwargs["request_pieces"][2].sequence == 1


def test_insert_prompt_memories_inserts_embedding(
    duckdb_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):

    request = PromptRequestResponse(request_pieces=[sample_conversations[0]])

    embedding_mock = MagicMock()
    embedding_mock.generate_text_embedding.returns = [0, 1, 2]
    duckdb_instance.enable_embedding(embedding_model=embedding_mock)

    with (
        patch("pyrit.memory.duckdb_memory.DuckDBMemory.add_request_pieces_to_memory"),
        patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_embeddings_to_memory") as mock_embedding,
    ):

        duckdb_instance.add_request_response_to_memory(request=request)

        assert mock_embedding.called
        assert embedding_mock.generate_text_embedding.called


def test_insert_prompt_memories_not_inserts_embedding(
    duckdb_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):

    request = PromptRequestResponse(request_pieces=[sample_conversations[0]])

    embedding_mock = MagicMock()
    embedding_mock.generate_text_embedding.returns = [0, 1, 2]
    duckdb_instance.enable_embedding(embedding_model=embedding_mock)
    duckdb_instance.disable_embedding()

    with (
        patch("pyrit.memory.duckdb_memory.DuckDBMemory.add_request_pieces_to_memory"),
        patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_embeddings_to_memory") as mock_embedding,
    ):

        duckdb_instance.add_request_response_to_memory(request=request)

        assert mock_embedding.assert_not_called


def test_export_conversation_by_orchestrator_id_file_created(
    duckdb_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):
    orchestrator1_id = sample_conversations[0].orchestrator_identifier["id"]

    # Default path in export_conversations()
    file_name = f"{orchestrator1_id}.json"
    file_path = Path(DB_DATA_PATH, file_name)

    duckdb_instance.exporter = MemoryExporter()

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_request_pieces") as mock_get:
        mock_get.return_value = sample_conversations
        duckdb_instance.export_conversations(orchestrator_id=orchestrator1_id, file_path=file_path)

        # Verify file was created
        assert file_path.exists()


def test_get_scores_by_orchestrator_id(
    duckdb_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):
    # create list of scores that are associated with sample conversation entries
    # assert that that list of scores is the same as expected :-)

    prompt_id = sample_conversations[0].id

    duckdb_instance.add_request_pieces_to_memory(request_pieces=sample_conversations)

    score = Score(
        score_value=str(0.8),
        score_value_description="High score",
        score_type="float_scale",
        score_category="test",
        score_rationale="Test score",
        score_metadata="Test metadata",
        scorer_class_identifier={"__type__": "TestScorer"},
        prompt_request_response_id=prompt_id,
    )

    duckdb_instance.add_scores_to_memory(scores=[score])

    # Fetch the score we just added
    db_score = duckdb_instance.get_scores_by_orchestrator_id(
        orchestrator_id=sample_conversations[0].orchestrator_identifier["id"]
    )

    assert len(db_score) == 1
    assert db_score[0].score_value == score.score_value
    assert db_score[0].score_value_description == score.score_value_description
    assert db_score[0].score_type == score.score_type
    assert db_score[0].score_category == score.score_category
    assert db_score[0].score_rationale == score.score_rationale
    assert db_score[0].score_metadata == score.score_metadata
    assert db_score[0].scorer_class_identifier == score.scorer_class_identifier
    assert db_score[0].prompt_request_response_id == score.prompt_request_response_id


@pytest.mark.parametrize("score_type", ["float_scale", "true_false"])
def test_add_score_get_score(
    duckdb_instance: MemoryInterface,
    sample_conversation_entries: Sequence[PromptMemoryEntry],
    score_type: Literal["float_scale"] | Literal["true_false"],
):
    prompt_id = sample_conversation_entries[0].id

    duckdb_instance._insert_entries(entries=sample_conversation_entries)

    score_value = str(True) if score_type == "true_false" else "0.8"

    score = Score(
        score_value=score_value,
        score_value_description="High score",
        score_type=score_type,
        score_category="test",
        score_rationale="Test score",
        score_metadata="Test metadata",
        scorer_class_identifier={"__type__": "TestScorer"},
        prompt_request_response_id=prompt_id,
    )

    duckdb_instance.add_scores_to_memory(scores=[score])

    # Fetch the score we just added
    db_score = duckdb_instance.get_scores_by_prompt_ids(prompt_request_response_ids=[prompt_id])
    assert db_score
    assert len(db_score) == 1
    assert db_score[0].score_value == score_value
    assert db_score[0].score_value_description == "High score"
    assert db_score[0].score_type == score_type
    assert db_score[0].score_category == "test"
    assert db_score[0].score_rationale == "Test score"
    assert db_score[0].score_metadata == "Test metadata"
    assert db_score[0].scorer_class_identifier == {"__type__": "TestScorer"}
    assert db_score[0].prompt_request_response_id == prompt_id


def test_add_score_duplicate_prompt(duckdb_instance: MemoryInterface):
    # Ensure that scores of duplicate prompts are linked back to the original
    original_id = uuid4()
    orchestrator = Orchestrator()
    conversation_id = str(uuid4())
    pieces = [
        PromptRequestPiece(
            id=original_id,
            role="assistant",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id,
            sequence=0,
            orchestrator_identifier=orchestrator.get_identifier(),
        )
    ]
    new_orchestrator_id = str(uuid4())
    duckdb_instance.add_request_pieces_to_memory(request_pieces=pieces)
    duckdb_instance.duplicate_conversation(new_orchestrator_id=new_orchestrator_id, conversation_id=conversation_id)
    dupe_piece = duckdb_instance.get_prompt_request_pieces(orchestrator_id=new_orchestrator_id)[0]
    dupe_id = dupe_piece.id

    score_id = uuid4()
    # score with prompt_request_response_id as dupe_id
    score = Score(
        id=score_id,
        score_value=str(0.8),
        score_value_description="High score",
        score_type="float_scale",
        score_category="test",
        score_rationale="Test score",
        score_metadata="Test metadata",
        scorer_class_identifier={"__type__": "TestScorer"},
        prompt_request_response_id=dupe_id,
    )
    duckdb_instance.add_scores_to_memory(scores=[score])

    assert score.prompt_request_response_id == original_id
    assert duckdb_instance.get_scores_by_prompt_ids(prompt_request_response_ids=[str(dupe_id)])[0].id == score_id
    assert duckdb_instance.get_scores_by_prompt_ids(prompt_request_response_ids=[str(original_id)])[0].id == score_id


def test_get_scores_by_memory_labels(duckdb_instance: MemoryInterface):
    prompt_id = uuid4()
    pieces = [
        PromptRequestPiece(
            id=prompt_id,
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            sequence=0,
            labels={"sample": "label"},
        )
    ]
    duckdb_instance.add_request_pieces_to_memory(request_pieces=pieces)

    score = Score(
        score_value=str(0.8),
        score_value_description="High score",
        score_type="float_scale",
        score_category="test",
        score_rationale="Test score",
        score_metadata="Test metadata",
        scorer_class_identifier={"__type__": "TestScorer"},
        prompt_request_response_id=prompt_id,
    )
    duckdb_instance.add_scores_to_memory(scores=[score])

    # Fetch the score we just added
    db_score = duckdb_instance.get_scores_by_memory_labels(memory_labels={"sample": "label"})

    assert len(db_score) == 1
    assert db_score[0].score_value == score.score_value
    assert db_score[0].score_value_description == score.score_value_description
    assert db_score[0].score_type == score.score_type
    assert db_score[0].score_category == score.score_category
    assert db_score[0].score_rationale == score.score_rationale
    assert db_score[0].score_metadata == score.score_metadata
    assert db_score[0].scorer_class_identifier == score.scorer_class_identifier
    assert db_score[0].prompt_request_response_id == prompt_id


@pytest.mark.asyncio
async def test_get_seed_prompts_no_filters(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts()
    assert len(result) == 2
    assert result[0].value == "prompt1"
    assert result[1].value == "prompt2"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_audio(duckdb_instance: MemoryInterface):
    """Test adding and retrieving seed prompts with an audio file."""
    temp_files = []
    try:
        # Create a temporary audio file
        audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_file.write(b"dummy audio content")
        audio_file.close()
        temp_files.append(audio_file.name)

        # Create a seed prompt for the audio file
        audio_prompt = SeedPrompt(value=audio_file.name, dataset_name="dataset_audio", data_type="audio_path")

        # Add seed prompt to memory
        await duckdb_instance.add_seed_prompts_to_memory_async(prompts=[audio_prompt], added_by="test_audio")

        # Retrieve and verify the seed prompts
        result = duckdb_instance.get_seed_prompts()
        assert len(result) == 1
        assert result[0].value.endswith(".wav")
        assert result[0].data_type == "audio_path"

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)


@pytest.mark.asyncio
async def test_get_seed_prompts_with_video(duckdb_instance: MemoryInterface):
    """Test adding and retrieving seed prompts with a video file."""
    temp_files = []
    try:
        # Create a temporary video file
        video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_file.write(b"dummy video content")
        video_file.close()
        temp_files.append(video_file.name)

        # Create a seed prompt for the video file
        video_prompt = SeedPrompt(value=video_file.name, dataset_name="dataset_video", data_type="video_path")

        # Add seed prompt to memory
        await duckdb_instance.add_seed_prompts_to_memory_async(prompts=[video_prompt], added_by="test_video")

        # Retrieve and verify the seed prompts
        result = duckdb_instance.get_seed_prompts()
        assert len(result) == 1
        assert result[0].value.endswith(".mp4")
        assert result[0].data_type == "video_path"

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)


@pytest.mark.asyncio
async def test_get_seed_prompts_with_image(duckdb_instance: MemoryInterface):
    """Test adding and retrieving seed prompts with an image file."""
    temp_files = []
    try:
        # Create a temporary image file
        image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_file.write(b"dummy image content")
        image_file.close()
        temp_files.append(image_file.name)

        # Create a seed prompt for the image file
        image_prompt = SeedPrompt(value=image_file.name, dataset_name="dataset_image", data_type="image_path")

        # Add seed prompt to memory
        await duckdb_instance.add_seed_prompts_to_memory_async(prompts=[image_prompt], added_by="test_image")

        # Retrieve and verify the seed prompts
        result = duckdb_instance.get_seed_prompts()
        assert len(result) == 1
        assert result[0].value.endswith(".png")
        assert result[0].data_type == "image_path"

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)


@pytest.mark.asyncio
async def test_get_seed_prompts_with_value_filter(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", data_type="text"),
        SeedPrompt(value="another prompt", dataset_name="dataset2", data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(value="prompt1")
    assert len(result) == 1
    assert result[0].value == "prompt1"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_dataset_name_filter(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(dataset_name="dataset1")
    assert len(result) == 1
    assert result[0].dataset_name == "dataset1"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_added_by_filter(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", added_by="user1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", added_by="user2", data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts)

    result = duckdb_instance.get_seed_prompts(added_by="user1")
    assert len(result) == 1
    assert result[0].added_by == "user1"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_source_filter(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", source="source1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", source="source2", data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(source="source1")
    assert len(result) == 1
    assert result[0].source == "source1"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_harm_categories_filter(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["category1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["category2"], data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(harm_categories=["category1"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_authors_filter(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", authors=["author2"], data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(authors=["author1"])
    assert len(result) == 1
    assert result[0].authors == ["author1"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_groups_filter(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", groups=["group1"], data_type="text"),
        SeedPrompt(value="prompt2", groups=["group2"], data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(groups=["group1"])
    assert len(result) == 1
    assert result[0].groups == ["group1"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_parameters_filter(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", parameters=["param1"], data_type="text"),
        SeedPrompt(value="prompt2", parameters=["param2"], data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(parameters=["param1"])
    assert len(result) == 1
    assert result[0].parameters == ["param1"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_metadata_filter(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", data_type="text", metadata={"key1": "value1", "key2": "value2"}),
        SeedPrompt(value="prompt2", data_type="text", metadata={"key1": "value2"}),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(metadata={"key1": "value1"})
    assert len(result) == 1
    assert result[0].metadata == {"key1": "value1", "key2": "value2"}


@pytest.mark.asyncio
async def test_get_seed_prompts_with_multiple_filters(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", added_by="user1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", added_by="user2", data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts)

    result = duckdb_instance.get_seed_prompts(dataset_name="dataset1", added_by="user1")
    assert len(result) == 1
    assert result[0].dataset_name == "dataset1"
    assert result[0].added_by == "user1"


@pytest.mark.asyncio
async def test_get_seed_prompts_with_empty_list_filters(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["harm1"], authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["harm2"], authors=["author2"], data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(harm_categories=[], authors=[])
    assert len(result) == 2


@pytest.mark.asyncio
async def test_get_seed_prompts_with_single_element_list_filters(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["category1"], authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["category2"], authors=["author2"], data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(harm_categories=["category1"], authors=["author1"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1"]
    assert result[0].authors == ["author1"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_multiple_elements_list_filters(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(
            value="prompt1",
            harm_categories=["category1", "category2"],
            authors=["author1", "author2"],
            data_type="text",
        ),
        SeedPrompt(value="prompt2", harm_categories=["category3"], authors=["author3"], data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(
        harm_categories=["category1", "category2"], authors=["author1", "author2"]
    )
    assert len(result) == 1
    assert result[0].harm_categories == ["category1", "category2"]
    assert result[0].authors == ["author1", "author2"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_multiple_elements_list_filters_additional(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(
            value="prompt1",
            harm_categories=["category1", "category2"],
            authors=["author1", "author2"],
            data_type="text",
        ),
        SeedPrompt(value="prompt2", harm_categories=["category3"], authors=["author3"], data_type="text"),
        SeedPrompt(
            value="prompt3",
            harm_categories=["category1", "category3"],
            authors=["author1", "author3"],
            data_type="text",
        ),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(
        harm_categories=["category1", "category3"], authors=["author1", "author3"]
    )
    assert len(result) == 1
    assert result[0].harm_categories == ["category1", "category3"]
    assert result[0].authors == ["author1", "author3"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_substring_filters_harm_categories(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["category1"], authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["category2"], authors=["author2"], data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(harm_categories=["ory1"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1"]

    result = duckdb_instance.get_seed_prompts(authors=["auth"])
    assert len(result) == 2
    assert result[0].authors == ["author1"]
    assert result[1].authors == ["author2"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_substring_filters_groups(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", groups=["group1"], data_type="text"),
        SeedPrompt(value="prompt2", groups=["group2"], data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(groups=["oup1"])
    assert len(result) == 1
    assert result[0].groups == ["group1"]

    result = duckdb_instance.get_seed_prompts(groups=["oup"])
    assert len(result) == 2
    assert result[0].groups == ["group1"]
    assert result[1].groups == ["group2"]


@pytest.mark.asyncio
async def test_get_seed_prompts_with_substring_filters_parameters(duckdb_instance: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", parameters=["param1"], data_type="text"),
        SeedPrompt(value="prompt2", parameters=["param2"], data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts, added_by="test")

    result = duckdb_instance.get_seed_prompts(parameters=["ram1"])
    assert len(result) == 1
    assert result[0].parameters == ["param1"]

    result = duckdb_instance.get_seed_prompts(parameters=["ram"])
    assert len(result) == 2
    assert result[0].parameters == ["param1"]
    assert result[1].parameters == ["param2"]


@pytest.mark.asyncio
async def test_add_seed_prompts_to_memory_empty_list(duckdb_instance: MemoryInterface):
    prompts: Sequence[SeedPrompt] = []
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=prompts, added_by="tester")
    stored_prompts = duckdb_instance.get_seed_prompts(dataset_name="test_dataset")
    assert len(stored_prompts) == 0


@pytest.mark.asyncio
async def test_add_seed_prompts_duplicate_entries_same_dataset(duckdb_instance: MemoryInterface):
    prompts: Sequence[SeedPrompt] = [
        SeedPrompt(value="prompt1", dataset_name="test_dataset", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="test_dataset", data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=prompts, added_by="tester")
    stored_prompts = duckdb_instance.get_seed_prompts(dataset_name="test_dataset")
    assert len(stored_prompts) == 2

    # Try to add prompt list with one duplicate prompt and one new prompt
    duplicate_prompts: Sequence[SeedPrompt] = [
        SeedPrompt(value="prompt1", dataset_name="test_dataset", data_type="text"),
        SeedPrompt(value="prompt3", dataset_name="test_dataset", data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=duplicate_prompts, added_by="tester")

    # Validate that only new prompt is added and the total prompt count is 3
    stored_prompts = duckdb_instance.get_seed_prompts(dataset_name="test_dataset")
    assert len(stored_prompts) == 3


@pytest.mark.asyncio
async def test_add_seed_prompts_duplicate_entries_different_datasets(duckdb_instance: MemoryInterface):
    prompts: Sequence[SeedPrompt] = [
        SeedPrompt(value="prompt1", dataset_name="test_dataset", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="test_dataset", data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=prompts, added_by="tester")
    stored_prompts = duckdb_instance.get_seed_prompts(dataset_name="test_dataset")
    assert len(stored_prompts) == 2

    # Try to add prompt list with one duplicate prompt and one new prompt
    duplicate_prompts: Sequence[SeedPrompt] = [
        SeedPrompt(value="prompt1", dataset_name="test_dataset2", data_type="text"),
        SeedPrompt(value="prompt3", dataset_name="test_dataset2", data_type="text"),
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=duplicate_prompts, added_by="tester")

    # Validate that only new prompt is added and the total prompt count is 3
    stored_prompts = duckdb_instance.get_seed_prompts()
    assert len(stored_prompts) == 4


def test_get_seed_prompt_dataset_names_empty(duckdb_instance: MemoryInterface):
    assert duckdb_instance.get_seed_prompt_dataset_names() == []


@pytest.mark.asyncio
async def test_get_seed_prompt_dataset_names_single(duckdb_instance: MemoryInterface):
    dataset_name = "test_dataset"
    seed_prompt = SeedPrompt(value="test_value", dataset_name=dataset_name, added_by="tester", data_type="text")
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=[seed_prompt])
    assert duckdb_instance.get_seed_prompt_dataset_names() == [dataset_name]


@pytest.mark.asyncio
async def test_get_seed_prompt_dataset_names_single_dataset_multiple_entries(duckdb_instance: MemoryInterface):
    dataset_name = "test_dataset"
    seed_prompt1 = SeedPrompt(value="test_value", dataset_name=dataset_name, added_by="tester", data_type="text")
    seed_prompt2 = SeedPrompt(value="test_value", dataset_name=dataset_name, added_by="tester", data_type="text")
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=[seed_prompt1, seed_prompt2])
    assert duckdb_instance.get_seed_prompt_dataset_names() == [dataset_name]


@pytest.mark.asyncio
async def test_get_seed_prompt_dataset_names_multiple(duckdb_instance: MemoryInterface):
    dataset_names = [f"dataset_{i}" for i in range(5)]
    seed_prompts = [
        SeedPrompt(value=f"value_{i}", dataset_name=dataset_name, added_by="tester", data_type="text")
        for i, dataset_name in enumerate(dataset_names)
    ]
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=seed_prompts)
    assert len(duckdb_instance.get_seed_prompt_dataset_names()) == 5
    assert sorted(duckdb_instance.get_seed_prompt_dataset_names()) == sorted(dataset_names)


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_empty_list(duckdb_instance: MemoryInterface):
    prompt_group = SeedPromptGroup(
        prompts=[SeedPrompt(value="Test prompt", added_by="tester", data_type="text", sequence=0)]
    )
    prompt_group.prompts = []
    with pytest.raises(ValueError, match="Prompt group must have at least one prompt."):
        await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_single_element(duckdb_instance: MemoryInterface):
    prompt = SeedPrompt(value="Test prompt", added_by="tester", data_type="text", sequence=0)
    prompt_group = SeedPromptGroup(prompts=[prompt])
    await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group], added_by="tester")
    assert len(duckdb_instance.get_seed_prompts()) == 1


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_multiple_elements(duckdb_instance: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0)
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1)
    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group], added_by="tester")
    assert len(duckdb_instance.get_seed_prompts()) == 2
    assert len(duckdb_instance.get_seed_prompt_groups()) == 1


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_no_elements(duckdb_instance: MemoryInterface):
    with pytest.raises(ValueError, match="SeedPromptGroup cannot be empty."):
        prompt_group = SeedPromptGroup(prompts=[])
        await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_single_element_no_added_by(duckdb_instance: MemoryInterface):
    prompt = SeedPrompt(value="Test prompt", data_type="text", sequence=0)
    prompt_group = SeedPromptGroup(prompts=[prompt])
    with pytest.raises(ValueError, match="The 'added_by' attribute must be set for each prompt."):
        await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_multiple_elements_no_added_by(duckdb_instance: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", data_type="text", sequence=0)
    prompt2 = SeedPrompt(value="Test prompt 2", data_type="text", sequence=1)
    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    with pytest.raises(ValueError, match="The 'added_by' attribute must be set for each prompt."):
        await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_inconsistent_group_ids(duckdb_instance: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0)
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1)

    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    prompt_group.prompts[0].prompt_group_id = uuid4()

    with pytest.raises(ValueError):
        await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_single_element_with_added_by(duckdb_instance: MemoryInterface):
    prompt = SeedPrompt(value="Test prompt", added_by="tester", data_type="text", sequence=0)
    prompt_group = SeedPromptGroup(prompts=[prompt])
    await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])
    assert len(duckdb_instance.get_seed_prompts()) == 1


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_multiple_elements_with_added_by(duckdb_instance: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0)
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1)
    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])
    assert len(duckdb_instance.get_seed_prompts()) == 2


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_multiple_groups_with_added_by(duckdb_instance: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0)
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1)
    prompt3 = SeedPrompt(value="Test prompt 3", added_by="tester", data_type="text", sequence=0)
    prompt4 = SeedPrompt(value="Test prompt 4", added_by="tester", data_type="text", sequence=1)

    prompt_group1 = SeedPromptGroup(prompts=[prompt1, prompt2])
    prompt_group2 = SeedPromptGroup(prompts=[prompt3, prompt4])

    await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group1, prompt_group2])
    assert len(duckdb_instance.get_seed_prompts()) == 4
    groups_from_memory = duckdb_instance.get_seed_prompt_groups()
    assert len(groups_from_memory) == 2
    assert groups_from_memory[0].prompts[0].id != groups_from_memory[1].prompts[1].id
    assert groups_from_memory[0].prompts[0].prompt_group_id == groups_from_memory[0].prompts[1].prompt_group_id
    assert groups_from_memory[1].prompts[0].prompt_group_id == groups_from_memory[1].prompts[1].prompt_group_id


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_with_all_modalities(duckdb_instance: MemoryInterface):
    """Test adding multiple prompt groups with different modalities using temporary files."""
    temp_files = []
    temp_dir = tempfile.TemporaryDirectory()
    duckdb_instance.results_path = temp_dir.name
    try:
        # Create a temporary image file
        image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_file.write(b"dummy image content")
        image_file.close()
        temp_files.append(image_file.name)

        # Create a temporary audio file
        audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_file.write(b"dummy audio content")
        audio_file.close()
        temp_files.append(audio_file.name)

        # Create a temporary video file
        video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_file.write(b"dummy video content")
        video_file.close()
        temp_files.append(video_file.name)

        # Create prompts with the temporary file paths
        prompt1 = SeedPrompt(value=image_file.name, added_by="testmultimodal", data_type="image_path", sequence=0)
        prompt2 = SeedPrompt(value=audio_file.name, added_by="testmultimodal", data_type="audio_path", sequence=1)
        prompt3 = SeedPrompt(value=video_file.name, added_by="testmultimodal", data_type="video_path", sequence=2)
        prompt4 = SeedPrompt(value="Test prompt 4", added_by="testmultimodal", data_type="text", sequence=3)

        # Create SeedPromptGroup
        seed_prompt_group1 = SeedPromptGroup(prompts=[prompt1, prompt2, prompt3, prompt4])

        # Add prompt groups to memory
        await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[seed_prompt_group1])

        # Assert the total number of prompts in memory
        assert len(duckdb_instance.get_seed_prompts(added_by="testmultimodal")) == 4

        # Retrieve and verify prompt groups from memory
        groups_from_memory = duckdb_instance.get_seed_prompt_groups(added_by="testmultimodal")
        assert len(groups_from_memory) == 1

        # Verify prompt group IDs are consistent within each group
        expected_prompt_group_id = groups_from_memory[0].prompts[0].prompt_group_id
        assert groups_from_memory[0].prompts[0].prompt_group_id == expected_prompt_group_id
        assert groups_from_memory[0].prompts[1].prompt_group_id == expected_prompt_group_id
        assert groups_from_memory[0].prompts[2].prompt_group_id == expected_prompt_group_id
        assert groups_from_memory[0].prompts[3].prompt_group_id == expected_prompt_group_id

        # Verify the specific data types and values
        assert groups_from_memory[0].prompts[0].data_type == "image_path"
        assert groups_from_memory[0].prompts[0].value.endswith(".png")
        assert groups_from_memory[0].prompts[1].data_type == "audio_path"
        assert groups_from_memory[0].prompts[1].value.endswith(".wav")
        assert groups_from_memory[0].prompts[2].data_type == "video_path"
        assert groups_from_memory[0].prompts[2].value.endswith(".mp4")
        assert groups_from_memory[0].prompts[3].data_type == "text"
        assert groups_from_memory[0].prompts[3].value == "Test prompt 4"

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        temp_dir.cleanup()


@pytest.mark.asyncio
async def test_add_seed_prompt_groups_to_memory_with_textimage_modalities(duckdb_instance: MemoryInterface):
    """Test adding multiple prompt groups with text and image modalities using temporary files."""
    temp_files = []
    temp_dir = tempfile.TemporaryDirectory()
    duckdb_instance.results_path = temp_dir.name
    try:
        # Create a temporary image file
        image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_file.write(b"dummy image content")
        image_file.close()
        temp_files.append(image_file.name)

        # Create prompts with the temporary file paths
        prompt1 = SeedPrompt(
            value=image_file.name, added_by="testtextimagemultimodal", data_type="image_path", sequence=0
        )
        prompt2 = SeedPrompt(value="Test prompt 2", added_by="testtextimagemultimodal", data_type="text", sequence=3)

        # Create SeedPromptGroup
        seed_prompt_group1 = SeedPromptGroup(prompts=[prompt1, prompt2])

        # Add prompt groups to memory
        await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[seed_prompt_group1])

        # Assert the total number of prompts in memory
        assert len(duckdb_instance.get_seed_prompts(added_by="testtextimagemultimodal")) == 2

        # Retrieve and verify prompt groups from memory
        groups_from_memory = duckdb_instance.get_seed_prompt_groups(added_by="testtextimagemultimodal")
        assert len(groups_from_memory) == 1

        # Verify prompt group IDs are consistent within each group
        expected_prompt_group_id = groups_from_memory[0].prompts[0].prompt_group_id
        assert groups_from_memory[0].prompts[0].prompt_group_id == expected_prompt_group_id
        assert groups_from_memory[0].prompts[1].prompt_group_id == expected_prompt_group_id

        # Verify the specific data types and values
        assert groups_from_memory[0].prompts[0].data_type == "image_path"
        assert groups_from_memory[0].prompts[0].value.endswith(".png")
        assert groups_from_memory[0].prompts[1].data_type == "text"
        assert groups_from_memory[0].prompts[1].value == "Test prompt 2"

    finally:
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        temp_dir.cleanup()


@pytest.mark.asyncio
async def test_get_seed_prompts_with_param_filters(duckdb_instance: MemoryInterface):
    template_value = "Test template {{ param1 }}"
    dataset_name = "dataset_1"
    harm_categories = ["category1"]
    added_by = "tester"
    parameters = ["param1"]
    template = SeedPrompt(
        value=template_value,
        dataset_name=dataset_name,
        parameters=parameters,
        harm_categories=harm_categories,
        added_by=added_by,
        data_type="text",
    )
    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=[template])

    templates = duckdb_instance.get_seed_prompts(
        value=template_value,
        dataset_name=dataset_name,
        harm_categories=harm_categories,
        added_by=added_by,
        parameters=parameters,
    )
    assert len(templates) == 1
    assert templates[0].value == template_value


def test_get_seed_prompt_groups_empty(duckdb_instance: MemoryInterface):
    assert duckdb_instance.get_seed_prompt_groups() == []


@pytest.mark.asyncio
async def test_get_seed_prompt_groups_with_dataset_name(duckdb_instance: MemoryInterface):
    dataset_name = "test_dataset"
    prompt_group = SeedPromptGroup(
        prompts=[
            SeedPrompt(value="Test prompt", dataset_name=dataset_name, added_by="tester", data_type="text", sequence=0)
        ]
    )
    await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])

    groups = duckdb_instance.get_seed_prompt_groups(dataset_name=dataset_name)
    assert len(groups) == 1
    assert groups[0].prompts[0].dataset_name == dataset_name


@pytest.mark.asyncio
async def test_get_seed_prompt_groups_with_multiple_filters(duckdb_instance: MemoryInterface):
    dataset_name = "dataset_1"
    data_types = ["text"]
    harm_categories = ["category1"]
    added_by = "tester"
    group = SeedPromptGroup(
        prompts=[
            SeedPrompt(
                value="Test prompt",
                dataset_name=dataset_name,
                harm_categories=harm_categories,
                added_by=added_by,
                sequence=0,
                data_type="text",
            )
        ]
    )
    await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[group])

    groups = duckdb_instance.get_seed_prompt_groups(
        dataset_name=dataset_name,
        data_types=data_types,
        harm_categories=harm_categories,
        added_by=added_by,
    )
    assert len(groups) == 1
    assert groups[0].prompts[0].dataset_name == dataset_name
    assert groups[0].prompts[0].added_by == added_by


@pytest.mark.asyncio
async def test_get_seed_prompt_groups_multiple_groups(duckdb_instance: MemoryInterface):
    group1 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 1", dataset_name="dataset_1", added_by="user1", sequence=0, data_type="text")]
    )
    group2 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 2", dataset_name="dataset_2", added_by="user2", sequence=0, data_type="text")]
    )
    await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[group1, group2])

    groups = duckdb_instance.get_seed_prompt_groups()
    assert len(groups) == 2


@pytest.mark.asyncio
async def test_get_seed_prompt_groups_multiple_groups_with_unique_ids(duckdb_instance: MemoryInterface):
    group1 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 1", dataset_name="dataset_1", added_by="user1", sequence=0, data_type="text")]
    )
    group2 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 2", dataset_name="dataset_2", added_by="user2", sequence=0, data_type="text")]
    )
    await duckdb_instance.add_seed_prompt_groups_to_memory(prompt_groups=[group1, group2])

    groups = duckdb_instance.get_seed_prompt_groups()
    assert len(groups) == 2
    # Check that each group has a unique prompt_group_id
    assert groups[0].prompts[0].prompt_group_id != groups[1].prompts[0].prompt_group_id


def test_export_all_conversations_file_created(duckdb_instance: MemoryInterface):
    duckdb_instance.exporter = MemoryExporter()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        with (
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_request_pieces") as mock_get_pieces,
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_scores_by_prompt_ids") as mock_get_scores,
        ):
            file_path = Path(temp_file.name)

            mock_get_pieces.return_value = [
                MagicMock(
                    original_prompt_id="1234",
                    converted_value="sample piece",
                    to_dict=lambda: {"prompt_request_response_id": "1234", "conversation": ["sample piece"]},
                )
            ]
            mock_get_scores.return_value = [
                MagicMock(
                    prompt_request_response_id="1234",
                    score_value=10,
                    to_dict=lambda: {"prompt_request_response_id": "1234", "score_value": 10},
                )
            ]

            assert file_path.exists()


def test_export_all_conversations_with_scores_correct_data(duckdb_instance: MemoryInterface):
    duckdb_instance.exporter = MemoryExporter()

    with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as temp_file:
        with (
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_request_pieces") as mock_get_pieces,
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_scores_by_prompt_ids") as mock_get_scores,
            patch.object(duckdb_instance.exporter, "export_data") as mock_export_data,
        ):
            file_path = Path(temp_file.name)

            mock_get_pieces.return_value = [MagicMock(original_prompt_id="1234", converted_value="sample piece")]
            mock_get_scores.return_value = [MagicMock(prompt_request_response_id="1234", score_value=10)]

            duckdb_instance.export_conversations(file_path=file_path)

            pos_arg, named_args = mock_export_data.call_args
            assert str(named_args["file_path"]) == temp_file.file.name
            assert str(named_args["export_type"]) == "json"
            assert pos_arg[0][0].original_prompt_id == "1234"
            assert pos_arg[0][0].converted_value == "sample piece"


def test_export_all_conversations_with_scores_empty_data(duckdb_instance: MemoryInterface):
    duckdb_instance.exporter = MemoryExporter()
    expected_data: Sequence = []
    with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as temp_file:
        with (
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_prompt_request_pieces") as mock_get_pieces,
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_scores_by_prompt_ids") as mock_get_scores,
            patch.object(duckdb_instance.exporter, "export_data") as mock_export_data,
        ):
            file_path = Path(temp_file.name)

            mock_get_pieces.return_value = []
            mock_get_scores.return_value = []

            duckdb_instance.export_conversations(file_path=file_path)
            mock_export_data.assert_called_once_with(expected_data, file_path=file_path, export_type="json")


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


@pytest.mark.asyncio
async def test_seed_prompt_hash_stored_and_retrieved(duckdb_instance: MemoryInterface):
    entries = [
        SeedPrompt(value="Hello 1", data_type="text"),
        SeedPrompt(value="Hello 2", data_type="text"),
    ]

    await duckdb_instance.add_seed_prompts_to_memory_async(prompts=entries, added_by="rlundeen")
    retrieved_entries = duckdb_instance.get_seed_prompts()

    assert len(retrieved_entries) == 2
    for prompt in retrieved_entries:
        assert prompt.value_sha256
