# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import uuid
from datetime import datetime
from typing import MutableSequence, Sequence
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.memory import MemoryInterface, PromptMemoryEntry
from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
)


def assert_original_value_in_list(original_value: str, prompt_request_pieces: Sequence[PromptRequestPiece]):
    for piece in prompt_request_pieces:
        if piece.original_value == original_value:
            return True
    raise AssertionError(f"Original value {original_value} not found in list")


def test_conversation_memory_empty_by_default(sqlite_instance: MemoryInterface):
    expected_count = 0
    c = sqlite_instance.get_prompt_request_pieces()
    assert len(c) == expected_count


@pytest.mark.parametrize("num_conversations", [1, 2, 3])
def test_add_request_pieces_to_memory(
    sqlite_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece], num_conversations: int
):
    for c in sample_conversations[:num_conversations]:
        c.conversation_id = sample_conversations[0].conversation_id
        c.role = sample_conversations[0].role
        c.sequence = 0

    request_response = PromptRequestResponse(request_pieces=sample_conversations[:num_conversations])

    sqlite_instance.add_request_response_to_memory(request=request_response)
    assert len(sqlite_instance.get_prompt_request_pieces()) == num_conversations


def test_get_prompt_request_pieces_uuid_and_string_ids(sqlite_instance: MemoryInterface):
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
    sqlite_instance.add_request_pieces_to_memory(request_pieces=pieces)

    uuid_results = sqlite_instance.get_prompt_request_pieces(prompt_ids=[uuid1, uuid2])
    assert len(uuid_results) == 2
    assert {str(uuid1), str(uuid2)} == {str(piece.id) for piece in uuid_results}

    str_results = sqlite_instance.get_prompt_request_pieces(prompt_ids=[str(uuid1), str(uuid2)])
    assert len(str_results) == 2
    assert {str(uuid1), str(uuid2)} == {str(piece.id) for piece in str_results}

    mixed_types: Sequence[str | uuid.UUID] = [uuid1, str(uuid2)]
    mixed_results = sqlite_instance.get_prompt_request_pieces(prompt_ids=mixed_types)
    assert len(mixed_results) == 2
    assert {str(uuid1), str(uuid2)} == {str(piece.id) for piece in mixed_results}

    single_uuid_result = sqlite_instance.get_prompt_request_pieces(prompt_ids=[uuid3])
    assert len(single_uuid_result) == 1
    assert str(single_uuid_result[0].id) == str(uuid3)

    single_str_result = sqlite_instance.get_prompt_request_pieces(prompt_ids=[str(uuid3)])
    assert len(single_str_result) == 1
    assert str(single_str_result[0].id) == str(uuid3)


def test_duplicate_memory(sqlite_instance: MemoryInterface):
    attack1 = PromptSendingAttack(objective_target=MagicMock())
    attack2 = PromptSendingAttack(objective_target=MagicMock())
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
            attack_identifier=attack1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id_1,
            sequence=0,
            attack_identifier=attack1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id_3,
            attack_identifier=attack2.get_identifier(),
        ),
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id_2,
            sequence=0,
            attack_identifier=attack1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id_2,
            sequence=0,
            attack_identifier=attack1.get_identifier(),
        ),
    ]
    sqlite_instance.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(sqlite_instance.get_prompt_request_pieces()) == 5
    attack3 = PromptSendingAttack(objective_target=MagicMock())
    new_conversation_id1 = sqlite_instance.duplicate_conversation(
        new_attack_id=attack3.get_identifier()["id"],
        conversation_id=conversation_id_1,
    )
    new_conversation_id2 = sqlite_instance.duplicate_conversation(
        new_attack_id=attack3.get_identifier()["id"],
        conversation_id=conversation_id_2,
    )
    all_pieces = sqlite_instance.get_prompt_request_pieces()
    assert len(all_pieces) == 9
    assert len([p for p in all_pieces if p.attack_identifier["id"] == attack1.get_identifier()["id"]]) == 4
    assert len([p for p in all_pieces if p.attack_identifier["id"] == attack2.get_identifier()["id"]]) == 1
    assert len([p for p in all_pieces if p.attack_identifier["id"] == attack3.get_identifier()["id"]]) == 4
    assert len([p for p in all_pieces if p.conversation_id == conversation_id_1]) == 2
    assert len([p for p in all_pieces if p.conversation_id == conversation_id_2]) == 2
    assert len([p for p in all_pieces if p.conversation_id == conversation_id_3]) == 1
    assert len([p for p in all_pieces if p.conversation_id == new_conversation_id1]) == 2
    assert len([p for p in all_pieces if p.conversation_id == new_conversation_id2]) == 2


# Ensure that the score entries are not duplicated when a conversation is duplicated
def test_duplicate_conversation_pieces_not_score(sqlite_instance: MemoryInterface):
    conversation_id = str(uuid4())
    prompt_id_1 = uuid4()
    prompt_id_2 = uuid4()
    attack1 = PromptSendingAttack(objective_target=MagicMock())
    memory_labels = {"sample": "label"}
    pieces = [
        PromptRequestPiece(
            id=prompt_id_1,
            role="assistant",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id,
            sequence=0,
            attack_identifier=attack1.get_identifier(),
            labels=memory_labels,
        ),
        PromptRequestPiece(
            id=prompt_id_2,
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id,
            sequence=0,
            attack_identifier=attack1.get_identifier(),
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
    sqlite_instance.add_request_pieces_to_memory(request_pieces=pieces)
    sqlite_instance.add_scores_to_memory(scores=scores)
    attack2 = PromptSendingAttack(objective_target=MagicMock())
    new_conversation_id = sqlite_instance.duplicate_conversation(
        new_attack_id=attack2.get_identifier()["id"],
        conversation_id=conversation_id,
    )
    new_pieces = sqlite_instance.get_prompt_request_pieces(conversation_id=new_conversation_id)
    new_pieces_ids = [str(p.id) for p in new_pieces]
    assert len(new_pieces) == 2
    original_ids = {piece.original_prompt_id for piece in new_pieces}
    assert original_ids == {prompt_id_1, prompt_id_2}

    for piece in new_pieces:
        assert piece.id not in (prompt_id_1, prompt_id_2)
    assert len(sqlite_instance.get_prompt_scores(labels=memory_labels)) == 2
    assert len(sqlite_instance.get_prompt_scores(attack_id=attack1.get_identifier()["id"])) == 2
    assert len(sqlite_instance.get_prompt_scores(attack_id=attack2.get_identifier()["id"])) == 2

    # The duplicate prompts ids should not have scores so only two scores are returned
    assert len(sqlite_instance.get_prompt_scores(prompt_ids=[str(prompt_id_1), str(prompt_id_2)] + new_pieces_ids)) == 2


def test_duplicate_conversation_excluding_last_turn(sqlite_instance: MemoryInterface):
    attack1 = PromptSendingAttack(objective_target=MagicMock())
    attack2 = PromptSendingAttack(objective_target=MagicMock())
    conversation_id_1 = "11111"
    conversation_id_2 = "22222"
    pieces = [
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=0,
            attack_identifier=attack1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=1,
            attack_identifier=attack1.get_identifier(),
        ),
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            sequence=2,
            conversation_id=conversation_id_1,
            attack_identifier=attack2.get_identifier(),
        ),
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id_2,
            sequence=2,
            attack_identifier=attack2.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id_2,
            sequence=3,
            attack_identifier=attack1.get_identifier(),
        ),
    ]
    sqlite_instance.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(sqlite_instance.get_prompt_request_pieces()) == 5
    attack3 = PromptSendingAttack(objective_target=MagicMock())

    new_conversation_id1 = sqlite_instance.duplicate_conversation_excluding_last_turn(
        new_attack_id=attack3.get_identifier()["id"],
        conversation_id=conversation_id_1,
    )

    all_memory = sqlite_instance.get_prompt_request_pieces()
    assert len(all_memory) == 7

    duplicate_conversation = sqlite_instance.get_prompt_request_pieces(conversation_id=new_conversation_id1)
    assert len(duplicate_conversation) == 2

    for piece in duplicate_conversation:
        assert piece.sequence < 2


def test_duplicate_conversation_excluding_last_turn_not_score(sqlite_instance: MemoryInterface):
    conversation_id = str(uuid4())
    prompt_id_1 = uuid4()
    prompt_id_2 = uuid4()
    attack1 = PromptSendingAttack(objective_target=MagicMock())
    memory_labels = {"sample": "label"}
    pieces = [
        PromptRequestPiece(
            id=prompt_id_1,
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id,
            sequence=0,
            attack_identifier=attack1.get_identifier(),
            labels=memory_labels,
        ),
        PromptRequestPiece(
            id=prompt_id_2,
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_id,
            sequence=1,
            attack_identifier=attack1.get_identifier(),
            labels=memory_labels,
        ),
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="That's good.",
            conversation_id=conversation_id,
            sequence=2,
            attack_identifier=attack1.get_identifier(),
            labels=memory_labels,
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="Thanks.",
            conversation_id=conversation_id,
            sequence=3,
            attack_identifier=attack1.get_identifier(),
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
    sqlite_instance.add_request_pieces_to_memory(request_pieces=pieces)
    sqlite_instance.add_scores_to_memory(scores=scores)
    attack2 = PromptSendingAttack(objective_target=MagicMock())

    new_conversation_id = sqlite_instance.duplicate_conversation_excluding_last_turn(
        new_attack_id=attack2.get_identifier()["id"],
        conversation_id=conversation_id,
    )
    new_pieces = sqlite_instance.get_prompt_request_pieces(conversation_id=new_conversation_id)
    new_pieces_ids = [str(p.id) for p in new_pieces]
    assert len(new_pieces) == 2
    assert new_pieces[0].original_prompt_id == prompt_id_1
    assert new_pieces[1].original_prompt_id == prompt_id_2
    assert new_pieces[0].id != prompt_id_1
    assert new_pieces[1].id != prompt_id_2
    assert len(sqlite_instance.get_prompt_scores(labels=memory_labels)) == 2
    assert len(sqlite_instance.get_prompt_scores(attack_id=attack1.get_identifier()["id"])) == 2
    assert len(sqlite_instance.get_prompt_scores(attack_id=attack2.get_identifier()["id"])) == 2
    # The duplicate prompts ids should not have scores so only two scores are returned
    assert len(sqlite_instance.get_prompt_scores(prompt_ids=[str(prompt_id_1), str(prompt_id_2)] + new_pieces_ids)) == 2


def test_duplicate_conversation_excluding_last_turn_same_attack(sqlite_instance: MemoryInterface):
    attack1 = PromptSendingAttack(objective_target=MagicMock())
    conversation_id_1 = "11111"
    pieces = [
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=0,
            attack_identifier=attack1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=1,
            attack_identifier=attack1.get_identifier(),
        ),
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=2,
            attack_identifier=attack1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            conversation_id=conversation_id_1,
            sequence=3,
            attack_identifier=attack1.get_identifier(),
        ),
    ]
    sqlite_instance.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(sqlite_instance.get_prompt_request_pieces()) == 4

    new_conversation_id1 = sqlite_instance.duplicate_conversation_excluding_last_turn(
        conversation_id=conversation_id_1,
    )

    all_memory = sqlite_instance.get_prompt_request_pieces()
    assert len(all_memory) == 6

    duplicate_conversation = sqlite_instance.get_prompt_request_pieces(conversation_id=new_conversation_id1)
    assert len(duplicate_conversation) == 2

    for piece in duplicate_conversation:
        assert piece.sequence < 2


def test_duplicate_memory_attack_id_collision(sqlite_instance: MemoryInterface):
    attack1 = PromptSendingAttack(objective_target=MagicMock())
    conversation_id = "11111"
    pieces = [
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_id,
            sequence=0,
            attack_identifier=attack1.get_identifier(),
        ),
    ]
    sqlite_instance.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(sqlite_instance.get_prompt_request_pieces()) == 1
    with pytest.raises(ValueError):
        sqlite_instance.duplicate_conversation(
            new_attack_id=str(attack1.get_identifier()["id"]),
            conversation_id=conversation_id,
        )


def test_add_request_pieces_to_memory_calls_validate(sqlite_instance: MemoryInterface):
    request_response = MagicMock(PromptRequestResponse)
    request_response.request_pieces = [MagicMock(PromptRequestPiece)]
    with (
        patch("pyrit.memory.sqlite_memory.SQLiteMemory.add_request_pieces_to_memory"),
        patch("pyrit.memory.memory_interface.MemoryInterface._update_sequence"),
    ):
        sqlite_instance.add_request_response_to_memory(request=request_response)
    assert request_response.validate.called


def test_add_request_pieces_to_memory_updates_sequence(
    sqlite_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):
    for conversation in sample_conversations:
        conversation.conversation_id = sample_conversations[0].conversation_id
        conversation.role = sample_conversations[0].role
        conversation.sequence = 17

    with patch("pyrit.memory.sqlite_memory.SQLiteMemory.add_request_pieces_to_memory") as mock_add:
        sqlite_instance.add_request_response_to_memory(
            request=PromptRequestResponse(request_pieces=sample_conversations)
        )
        assert mock_add.called

        args, kwargs = mock_add.call_args
        assert kwargs["request_pieces"][0].sequence == 0, "Sequence should be reset to 0"
        assert kwargs["request_pieces"][1].sequence == 0, "Sequence should be reset to 0"
        assert kwargs["request_pieces"][2].sequence == 0, "Sequence should be reset to 0"


def test_add_request_pieces_to_memory_updates_sequence_with_prev_conversation(
    sqlite_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):

    for conversation in sample_conversations:
        conversation.conversation_id = sample_conversations[0].conversation_id
        conversation.role = sample_conversations[0].role
        conversation.sequence = 17

    # insert one of these into memory
    sqlite_instance.add_request_response_to_memory(request=PromptRequestResponse(request_pieces=sample_conversations))

    with patch("pyrit.memory.sqlite_memory.SQLiteMemory.add_request_pieces_to_memory") as mock_add:
        sqlite_instance.add_request_response_to_memory(
            request=PromptRequestResponse(request_pieces=sample_conversations)
        )
        assert mock_add.called

        args, kwargs = mock_add.call_args
        assert kwargs["request_pieces"][0].sequence == 1, "Sequence should increment previous conversation by 1"
        assert kwargs["request_pieces"][1].sequence == 1
        assert kwargs["request_pieces"][2].sequence == 1


def test_insert_prompt_memories_inserts_embedding(
    sqlite_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):

    request = PromptRequestResponse(request_pieces=[sample_conversations[0]])

    embedding_mock = MagicMock()
    embedding_mock.generate_text_embedding.returns = [0, 1, 2]
    sqlite_instance.enable_embedding(embedding_model=embedding_mock)

    with (
        patch("pyrit.memory.sqlite_memory.SQLiteMemory.add_request_pieces_to_memory"),
        patch("pyrit.memory.sqlite_memory.SQLiteMemory._add_embeddings_to_memory") as mock_embedding,
    ):

        sqlite_instance.add_request_response_to_memory(request=request)

        assert mock_embedding.called
        assert embedding_mock.generate_text_embedding.called


def test_insert_prompt_memories_not_inserts_embedding(
    sqlite_instance: MemoryInterface, sample_conversations: Sequence[PromptRequestPiece]
):

    request = PromptRequestResponse(request_pieces=[sample_conversations[0]])

    embedding_mock = MagicMock()
    embedding_mock.generate_text_embedding.returns = [0, 1, 2]
    sqlite_instance.enable_embedding(embedding_model=embedding_mock)
    sqlite_instance.disable_embedding()

    with (
        patch("pyrit.memory.sqlite_memory.SQLiteMemory.add_request_pieces_to_memory"),
        patch("pyrit.memory.sqlite_memory.SQLiteMemory._add_embeddings_to_memory") as mock_embedding,
    ):

        sqlite_instance.add_request_response_to_memory(request=request)

        assert mock_embedding.assert_not_called


def test_get_prompt_request_pieces_labels(sqlite_instance: MemoryInterface):
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

    sqlite_instance._insert_entries(entries=entries)

    retrieved_entries = sqlite_instance.get_prompt_request_pieces(labels=labels)

    assert len(retrieved_entries) == 2  # Two entries should have the specific memory labels
    for retrieved_entry in retrieved_entries:
        assert "op_name" in retrieved_entry.labels
        assert "user_name" in retrieved_entry.labels
        assert "harm_category" in retrieved_entry.labels


def test_get_prompt_request_pieces_metadata(sqlite_instance: MemoryInterface):
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

    sqlite_instance._insert_entries(entries=entries)

    retrieved_entries = sqlite_instance.get_prompt_request_pieces(prompt_metadata={"key2": "value2"})

    assert len(retrieved_entries) == 2  # Two entries should have the specific memory labels
    for retrieved_entry in retrieved_entries:
        assert "key2" in retrieved_entry.prompt_metadata


def test_get_prompt_request_pieces_id(sqlite_instance: MemoryInterface):
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

    sqlite_instance._insert_entries(entries=entries)

    retrieved_entries = sqlite_instance.get_prompt_request_pieces(prompt_ids=[id_1, id_2])

    assert len(retrieved_entries) == 2
    assert_original_value_in_list("Hello 1", retrieved_entries)
    assert_original_value_in_list("Hello 2", retrieved_entries)


def test_get_prompt_request_pieces_attack(sqlite_instance: MemoryInterface):

    attack1 = PromptSendingAttack(objective_target=MagicMock())
    attack2 = PromptSendingAttack(objective_target=MagicMock())

    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 1",
                attack_identifier=attack1.get_identifier(),
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="assistant",
                original_value="Hello 2",
                attack_identifier=attack2.get_identifier(),
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                role="user",
                original_value="Hello 3",
                attack_identifier=attack1.get_identifier(),
            )
        ),
    ]

    sqlite_instance._insert_entries(entries=entries)

    attack1_entries = sqlite_instance.get_prompt_request_pieces(attack_id=attack1.get_identifier()["id"])

    assert len(attack1_entries) == 2
    assert_original_value_in_list("Hello 1", attack1_entries)
    assert_original_value_in_list("Hello 3", attack1_entries)


def test_get_prompt_request_pieces_sent_after(sqlite_instance: MemoryInterface):
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

    sqlite_instance._insert_entries(entries=entries)

    retrieved_entries = sqlite_instance.get_prompt_request_pieces(sent_after=datetime(2024, 1, 1))

    assert len(retrieved_entries) == 1
    assert "Hello 3" in retrieved_entries[0].original_value


def test_get_prompt_request_pieces_sent_before(sqlite_instance: MemoryInterface):
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

    sqlite_instance._insert_entries(entries=entries)

    retrieved_entries = sqlite_instance.get_prompt_request_pieces(sent_before=datetime(2024, 1, 1))

    assert len(retrieved_entries) == 2
    assert_original_value_in_list("Hello 1", retrieved_entries)
    assert_original_value_in_list("Hello 2", retrieved_entries)


def test_get_prompt_request_pieces_by_value(sqlite_instance: MemoryInterface):
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

    sqlite_instance._insert_entries(entries=entries)
    retrieved_entries = sqlite_instance.get_prompt_request_pieces(converted_values=["Hello 2", "Hello 3"])

    assert len(retrieved_entries) == 2
    assert_original_value_in_list("Hello 2", retrieved_entries)
    assert_original_value_in_list("Hello 3", retrieved_entries)


def test_get_prompt_request_pieces_by_hash(sqlite_instance: MemoryInterface):
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

    sqlite_instance.add_request_pieces_to_memory(request_pieces=entries)
    retrieved_entries = sqlite_instance.get_prompt_request_pieces(converted_value_sha256=["hash1"])

    assert len(retrieved_entries) == 2
    assert_original_value_in_list("Hello 1", retrieved_entries)
    assert_original_value_in_list("Hello 2", retrieved_entries)


def test_get_prompt_request_pieces_with_non_matching_memory_labels(sqlite_instance: MemoryInterface):
    attack = PromptSendingAttack(objective_target=MagicMock())
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
                attack_identifier=attack.get_identifier(),
            )
        ),
    ]

    sqlite_instance._insert_entries(entries=entries)
    labels = {"nonexistent_key": "nonexiststent_value"}
    retrieved_entries = sqlite_instance.get_prompt_request_pieces(labels=labels)

    assert len(retrieved_entries) == 0  # zero entries found since invalid memory labels passed


def test_get_prompt_request_pieces_sorts(
    sqlite_instance: MemoryInterface, sample_conversations: MutableSequence[PromptRequestPiece]
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

    sqlite_instance.add_request_pieces_to_memory(request_pieces=sample_conversations)

    response = sqlite_instance.get_prompt_request_pieces()

    current_value = response[0].conversation_id
    for obj in response[1:]:
        new_value = obj.conversation_id
        if new_value != current_value:
            if any(o.conversation_id == current_value for o in response[response.index(obj) :]):
                assert False, "Conversation IDs are not grouped together"


def test_prompt_piece_scores_duplicate_piece(sqlite_instance: MemoryInterface):
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

    sqlite_instance.add_request_pieces_to_memory(request_pieces=pieces)

    score = Score(
        score_value=str(0.8),
        score_value_description="Sample description",
        score_type="float_scale",
        score_category="Sample category",
        score_rationale="Sample rationale",
        score_metadata="Sample metadata",
        prompt_request_response_id=original_id,
    )
    sqlite_instance.add_scores_to_memory(scores=[score])

    retrieved_pieces = sqlite_instance.get_prompt_request_pieces()

    assert len(retrieved_pieces[0].scores) == 1
    assert retrieved_pieces[0].scores[0].score_value == "0.8"

    # Check that the duplicate piece has the same score as the original
    assert len(retrieved_pieces[1].scores) == 1
    assert retrieved_pieces[1].scores[0].score_value == "0.8"


@pytest.mark.asyncio
async def test_prompt_piece_hash_stored_and_retrieved(sqlite_instance: MemoryInterface):
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

    sqlite_instance.add_request_pieces_to_memory(request_pieces=entries)
    retrieved_entries = sqlite_instance.get_prompt_request_pieces()

    assert len(retrieved_entries) == 2
    for prompt in retrieved_entries:
        assert prompt.converted_value_sha256
        assert prompt.original_value_sha256


@pytest.mark.asyncio
async def test_seed_prompt_hash_stored_and_retrieved(sqlite_instance: MemoryInterface):
    """Test that seed prompt hash values are properly stored and retrieved."""
    from pyrit.models import SeedPrompt

    # Create a seed prompt
    seed_prompt = SeedPrompt(
        value="Test seed prompt",
        data_type="text",
        dataset_name="test_dataset",
        added_by="test_user",
    )

    # Add to memory
    await sqlite_instance.add_seed_prompts_to_memory_async(prompts=[seed_prompt])

    # Retrieve and verify hash
    retrieved_prompts = sqlite_instance.get_seed_prompts(value_sha256=[seed_prompt.value_sha256])
    assert len(retrieved_prompts) == 1
    assert retrieved_prompts[0].value_sha256 == seed_prompt.value_sha256
