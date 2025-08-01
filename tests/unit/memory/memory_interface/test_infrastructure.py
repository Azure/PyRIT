# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import Sequence
from unittest.mock import MagicMock, patch

import pytest
from uuid import uuid4

from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.orchestrator import Orchestrator


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