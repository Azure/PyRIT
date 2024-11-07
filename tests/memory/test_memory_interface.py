# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
from typing import Generator, Literal
from unittest.mock import MagicMock, patch
from pathlib import Path
from uuid import uuid4
import pytest
import random
from string import ascii_lowercase

from pyrit.common.path import RESULTS_PATH
from pyrit.memory import MemoryInterface, MemoryExporter, PromptMemoryEntry
from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.orchestrator import Orchestrator

from tests.mocks import get_memory_interface, get_sample_conversations, get_sample_conversation_entries


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def sample_conversation_entries() -> list[PromptMemoryEntry]:
    return get_sample_conversation_entries()


def generate_random_string(length: int = 10) -> str:
    return "".join(random.choice(ascii_lowercase) for _ in range(length))


def test_memory(memory: MemoryInterface):
    assert memory


def test_conversation_memory_empty_by_default(memory: MemoryInterface):
    expected_count = 0
    c = memory.get_all_prompt_pieces()
    assert len(c) == expected_count


@pytest.mark.parametrize("num_conversations", [1, 2, 3])
def test_add_request_pieces_to_memory(
    memory: MemoryInterface, sample_conversations: list[PromptRequestPiece], num_conversations: int
):
    for c in sample_conversations[:num_conversations]:
        c.conversation_id = sample_conversations[0].conversation_id
        c.role = sample_conversations[0].role

    request_response = PromptRequestResponse(request_pieces=sample_conversations[:num_conversations])

    memory.add_request_response_to_memory(request=request_response)
    assert len(memory.get_all_prompt_pieces()) == num_conversations


def test_duplicate_memory(memory: MemoryInterface):
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
    memory.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(memory.get_all_prompt_pieces()) == 5
    orchestrator3 = Orchestrator()
    new_conversation_id1 = memory.duplicate_conversation_for_new_orchestrator(
        new_orchestrator_id=orchestrator3.get_identifier()["id"],
        conversation_id=conversation_id_1,
    )
    new_conversation_id2 = memory.duplicate_conversation_for_new_orchestrator(
        new_orchestrator_id=orchestrator3.get_identifier()["id"],
        conversation_id=conversation_id_2,
    )
    all_pieces = memory.get_all_prompt_pieces()
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
def test_duplicate_conversation_pieces_not_score(memory: MemoryInterface):
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
    memory.add_request_pieces_to_memory(request_pieces=pieces)
    memory.add_scores_to_memory(scores=scores)
    orchestrator2 = Orchestrator()
    new_conversation_id = memory.duplicate_conversation_for_new_orchestrator(
        new_orchestrator_id=orchestrator2.get_identifier()["id"],
        conversation_id=conversation_id,
    )
    new_pieces = memory._get_prompt_pieces_with_conversation_id(conversation_id=new_conversation_id)
    new_pieces_ids = [str(p.id) for p in new_pieces]
    assert len(new_pieces) == 2
    assert new_pieces[0].original_prompt_id == prompt_id_1
    assert new_pieces[1].original_prompt_id == prompt_id_2
    assert new_pieces[0].id != prompt_id_1
    assert new_pieces[1].id != prompt_id_2
    assert len(memory.get_scores_by_memory_labels(memory_labels=memory_labels)) == 2
    assert len(memory.get_scores_by_orchestrator_id(orchestrator_id=orchestrator1.get_identifier()["id"])) == 2
    assert len(memory.get_scores_by_orchestrator_id(orchestrator_id=orchestrator2.get_identifier()["id"])) == 2
    # The duplicate prompts ids should not have scores so only two scores are returned
    assert (
        len(
            memory.get_scores_by_prompt_ids(
                prompt_request_response_ids=[str(prompt_id_1), str(prompt_id_2)] + new_pieces_ids
            )
        )
        == 2
    )


def test_duplicate_conversation_excluding_last_turn(memory: MemoryInterface):
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
    memory.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(memory.get_all_prompt_pieces()) == 5
    orchestrator3 = Orchestrator()

    new_conversation_id1 = memory.duplicate_conversation_excluding_last_turn(
        new_orchestrator_id=orchestrator3.get_identifier()["id"],
        conversation_id=conversation_id_1,
    )

    all_memory = memory.get_all_prompt_pieces()
    assert len(all_memory) == 7

    duplicate_conversation = memory._get_prompt_pieces_with_conversation_id(conversation_id=new_conversation_id1)
    assert len(duplicate_conversation) == 2

    for piece in duplicate_conversation:
        assert piece.sequence < 2


def test_duplicate_conversation_excluding_last_turn_not_score(memory: MemoryInterface):
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
    memory.add_request_pieces_to_memory(request_pieces=pieces)
    memory.add_scores_to_memory(scores=scores)
    orchestrator2 = Orchestrator()

    new_conversation_id = memory.duplicate_conversation_excluding_last_turn(
        new_orchestrator_id=orchestrator2.get_identifier()["id"],
        conversation_id=conversation_id,
    )
    new_pieces = memory._get_prompt_pieces_with_conversation_id(conversation_id=new_conversation_id)
    new_pieces_ids = [str(p.id) for p in new_pieces]
    assert len(new_pieces) == 2
    assert new_pieces[0].original_prompt_id == prompt_id_1
    assert new_pieces[1].original_prompt_id == prompt_id_2
    assert new_pieces[0].id != prompt_id_1
    assert new_pieces[1].id != prompt_id_2
    assert len(memory.get_scores_by_memory_labels(memory_labels=memory_labels)) == 2
    assert len(memory.get_scores_by_orchestrator_id(orchestrator_id=orchestrator1.get_identifier()["id"])) == 2
    assert len(memory.get_scores_by_orchestrator_id(orchestrator_id=orchestrator2.get_identifier()["id"])) == 2
    # The duplicate prompts ids should not have scores so only two scores are returned
    assert (
        len(
            memory.get_scores_by_prompt_ids(
                prompt_request_response_ids=[str(prompt_id_1), str(prompt_id_2)] + new_pieces_ids
            )
        )
        == 2
    )


def test_duplicate_conversation_excluding_last_turn_same_orchestrator(memory: MemoryInterface):
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
    memory.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(memory.get_all_prompt_pieces()) == 4

    new_conversation_id1 = memory.duplicate_conversation_excluding_last_turn(
        conversation_id=conversation_id_1,
    )

    all_memory = memory.get_all_prompt_pieces()
    assert len(all_memory) == 6

    duplicate_conversation = memory._get_prompt_pieces_with_conversation_id(conversation_id=new_conversation_id1)
    assert len(duplicate_conversation) == 2

    for piece in duplicate_conversation:
        assert piece.sequence < 2


def test_duplicate_memory_orchestrator_id_collision(memory: MemoryInterface):
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
    memory.add_request_pieces_to_memory(request_pieces=pieces)
    assert len(memory.get_all_prompt_pieces()) == 1
    with pytest.raises(ValueError):
        memory.duplicate_conversation_for_new_orchestrator(
            new_orchestrator_id=str(orchestrator1.get_identifier()["id"]),
            conversation_id=conversation_id,
        )


def test_add_request_pieces_to_memory_calls_validate(memory: MemoryInterface):
    request_response = MagicMock(PromptRequestResponse)
    request_response.request_pieces = [MagicMock(PromptRequestPiece)]
    with (
        patch("pyrit.memory.duckdb_memory.DuckDBMemory.add_request_pieces_to_memory"),
        patch("pyrit.memory.memory_interface.MemoryInterface._update_sequence"),
    ):
        memory.add_request_response_to_memory(request=request_response)
    assert request_response.validate.called


def test_add_request_pieces_to_memory_updates_sequence(
    memory: MemoryInterface, sample_conversations: list[PromptRequestPiece]
):
    for conversation in sample_conversations:
        conversation.conversation_id = sample_conversations[0].conversation_id
        conversation.role = sample_conversations[0].role
        conversation.sequence = 17

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory.add_request_pieces_to_memory") as mock_add:
        memory.add_request_response_to_memory(request=PromptRequestResponse(request_pieces=sample_conversations))
        assert mock_add.called

        args, kwargs = mock_add.call_args
        assert kwargs["request_pieces"][0].sequence == 0, "Sequence should be reset to 0"
        assert kwargs["request_pieces"][1].sequence == 0, "Sequence should be reset to 0"
        assert kwargs["request_pieces"][2].sequence == 0, "Sequence should be reset to 0"


def test_add_request_pieces_to_memory_updates_sequence_with_prev_conversation(
    memory: MemoryInterface, sample_conversations: list[PromptRequestPiece]
):

    for conversation in sample_conversations:
        conversation.conversation_id = sample_conversations[0].conversation_id
        conversation.role = sample_conversations[0].role
        conversation.sequence = 17

    # insert one of these into memory
    memory.add_request_response_to_memory(request=PromptRequestResponse(request_pieces=sample_conversations))

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory.add_request_pieces_to_memory") as mock_add:
        memory.add_request_response_to_memory(request=PromptRequestResponse(request_pieces=sample_conversations))
        assert mock_add.called

        args, kwargs = mock_add.call_args
        assert kwargs["request_pieces"][0].sequence == 1, "Sequence should increment previous conversation by 1"
        assert kwargs["request_pieces"][1].sequence == 1
        assert kwargs["request_pieces"][2].sequence == 1


def test_insert_prompt_memories_inserts_embedding(
    memory: MemoryInterface, sample_conversations: list[PromptRequestPiece]
):

    request = PromptRequestResponse(request_pieces=[sample_conversations[0]])

    embedding_mock = MagicMock()
    embedding_mock.generate_text_embedding.returns = [0, 1, 2]
    memory.enable_embedding(embedding_model=embedding_mock)

    with (
        patch("pyrit.memory.duckdb_memory.DuckDBMemory.add_request_pieces_to_memory"),
        patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_embeddings_to_memory") as mock_embedding,
    ):

        memory.add_request_response_to_memory(request=request)

        assert mock_embedding.called
        assert embedding_mock.generate_text_embedding.called


def test_insert_prompt_memories_not_inserts_embedding(
    memory: MemoryInterface, sample_conversations: list[PromptRequestPiece]
):

    request = PromptRequestResponse(request_pieces=[sample_conversations[0]])

    embedding_mock = MagicMock()
    embedding_mock.generate_text_embedding.returns = [0, 1, 2]
    memory.enable_embedding(embedding_model=embedding_mock)
    memory.disable_embedding()

    with (
        patch("pyrit.memory.duckdb_memory.DuckDBMemory.add_request_pieces_to_memory"),
        patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_embeddings_to_memory") as mock_embedding,
    ):

        memory.add_request_response_to_memory(request=request)

        assert mock_embedding.assert_not_called


def test_get_orchestrator_conversation_sorting(memory: MemoryInterface, sample_conversations: list[PromptRequestPiece]):
    conversation_id = sample_conversations[0].conversation_id

    # This new conversation piece should be grouped with other messages in the conversation
    sample_conversations.append(
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            conversation_id=conversation_id,
        )
    )

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory._get_prompt_pieces_by_orchestrator") as mock_get:

        mock_get.return_value = sample_conversations
        orchestrator_id = sample_conversations[0].orchestrator_identifier["id"]

        response = memory.get_prompt_request_piece_by_orchestrator_id(orchestrator_id=orchestrator_id)

        current_value = response[0].conversation_id
        for obj in response[1:]:
            new_value = obj.conversation_id
            if new_value != current_value:
                if any(o.conversation_id == current_value for o in response[response.index(obj) :]):
                    assert False, "Conversation IDs are not grouped together"


def test_export_conversation_by_orchestrator_id_file_created(
    memory: MemoryInterface, sample_conversation_entries: list[PromptMemoryEntry]
):
    orchestrator1_id = sample_conversation_entries[0].get_prompt_request_piece().orchestrator_identifier["id"]

    # Default path in export_conversation_by_orchestrator_id()
    file_name = f"{orchestrator1_id}.json"
    file_path = Path(RESULTS_PATH, file_name)

    memory.exporter = MemoryExporter()

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory._get_prompt_pieces_by_orchestrator") as mock_get:
        mock_get.return_value = sample_conversation_entries
        memory.export_conversation_by_orchestrator_id(orchestrator_id=orchestrator1_id)

        # Verify file was created
        assert file_path.exists()


def test_get_prompt_ids_by_orchestrator(memory: MemoryInterface, sample_conversation_entries: list[PromptMemoryEntry]):
    orchestrator1_id = sample_conversation_entries[0].get_prompt_request_piece().orchestrator_identifier["id"]

    sample_conversation_ids = []
    for entry in sample_conversation_entries:
        sample_conversation_ids.append(str(entry.get_prompt_request_piece().id))

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory._get_prompt_pieces_by_orchestrator") as mock_get:
        mock_get.return_value = sample_conversation_entries
        prompt_ids = memory.get_prompt_ids_by_orchestrator(orchestrator_id=orchestrator1_id)

        assert sample_conversation_ids == prompt_ids


def test_get_scores_by_orchestrator_id(memory: MemoryInterface, sample_conversations: list[PromptRequestPiece]):
    # create list of scores that are associated with sample conversation entries
    # assert that that list of scores is the same as expected :-)

    prompt_id = sample_conversations[0].id

    memory.add_request_pieces_to_memory(request_pieces=sample_conversations)

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

    memory.add_scores_to_memory(scores=[score])

    # Fetch the score we just added
    db_score = memory.get_scores_by_orchestrator_id(
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
    memory: MemoryInterface,
    sample_conversation_entries: list[PromptMemoryEntry],
    score_type: Literal["float_scale"] | Literal["true_false"],
):
    prompt_id = sample_conversation_entries[0].id

    memory.insert_entries(entries=sample_conversation_entries)

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

    memory.add_scores_to_memory(scores=[score])

    # Fetch the score we just added
    db_score = memory.get_scores_by_prompt_ids(prompt_request_response_ids=[prompt_id])
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


def test_add_score_duplicate_prompt(memory: MemoryInterface):
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
    memory.add_request_pieces_to_memory(request_pieces=pieces)
    memory.duplicate_conversation_for_new_orchestrator(
        new_orchestrator_id=new_orchestrator_id, conversation_id=conversation_id
    )
    dupe_piece = memory.get_prompt_request_piece_by_orchestrator_id(orchestrator_id=new_orchestrator_id)[0]
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
    memory.add_scores_to_memory(scores=[score])

    assert score.prompt_request_response_id == original_id
    assert memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(dupe_id)])[0].id == score_id
    assert memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(original_id)])[0].id == score_id


def test_get_scores_by_memory_labels(memory: MemoryInterface):
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
    memory.add_request_pieces_to_memory(request_pieces=pieces)

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
    memory.add_scores_to_memory(scores=[score])

    # Fetch the score we just added
    db_score = memory.get_scores_by_memory_labels(memory_labels={"sample": "label"})

    assert len(db_score) == 1
    assert db_score[0].score_value == score.score_value
    assert db_score[0].score_value_description == score.score_value_description
    assert db_score[0].score_type == score.score_type
    assert db_score[0].score_category == score.score_category
    assert db_score[0].score_rationale == score.score_rationale
    assert db_score[0].score_metadata == score.score_metadata
    assert db_score[0].scorer_class_identifier == score.scorer_class_identifier
    assert db_score[0].prompt_request_response_id == prompt_id


def test_get_seed_prompts_no_filters(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts()
    assert len(result) == 2
    assert result[0].value == "prompt1"
    assert result[1].value == "prompt2"


def test_get_seed_prompts_with_value_filter(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", data_type="text"),
        SeedPrompt(value="another prompt", dataset_name="dataset2", data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(value="prompt1")
    assert len(result) == 1
    assert result[0].value == "prompt1"


def test_get_seed_prompts_with_dataset_name_filter(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(dataset_name="dataset1")
    assert len(result) == 1
    assert result[0].dataset_name == "dataset1"


def test_get_seed_prompts_with_added_by_filter(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", added_by="user1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", added_by="user2", data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts)

    result = memory.get_seed_prompts(added_by="user1")
    assert len(result) == 1
    assert result[0].added_by == "user1"


def test_get_seed_prompts_with_source_filter(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", source="source1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", source="source2", data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(source="source1")
    assert len(result) == 1
    assert result[0].source == "source1"


def test_get_seed_prompts_with_harm_categories_filter(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["category1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["category2"], data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(harm_categories=["category1"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1"]


def test_get_seed_prompts_with_authors_filter(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", authors=["author2"], data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(authors=["author1"])
    assert len(result) == 1
    assert result[0].authors == ["author1"]


def test_get_seed_prompts_with_groups_filter(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", groups=["group1"], data_type="text"),
        SeedPrompt(value="prompt2", groups=["group2"], data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(groups=["group1"])
    assert len(result) == 1
    assert result[0].groups == ["group1"]


def test_get_seed_prompts_with_parameters_filter(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", parameters=["param1"], data_type="text"),
        SeedPrompt(value="prompt2", parameters=["param2"], data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(parameters=["param1"])
    assert len(result) == 1
    assert result[0].parameters == ["param1"]


def test_get_seed_prompts_with_multiple_filters(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", dataset_name="dataset1", added_by="user1", data_type="text"),
        SeedPrompt(value="prompt2", dataset_name="dataset2", added_by="user2", data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts)

    result = memory.get_seed_prompts(dataset_name="dataset1", added_by="user1")
    assert len(result) == 1
    assert result[0].dataset_name == "dataset1"
    assert result[0].added_by == "user1"


def test_get_seed_prompts_with_empty_list_filters(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["harm1"], authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["harm2"], authors=["author2"], data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(harm_categories=[], authors=[])
    assert len(result) == 2


def test_get_seed_prompts_with_single_element_list_filters(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["category1"], authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["category2"], authors=["author2"], data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(harm_categories=["category1"], authors=["author1"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1"]
    assert result[0].authors == ["author1"]


def test_get_seed_prompts_with_multiple_elements_list_filters(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(
            value="prompt1",
            harm_categories=["category1", "category2"],
            authors=["author1", "author2"],
            data_type="text",
        ),
        SeedPrompt(value="prompt2", harm_categories=["category3"], authors=["author3"], data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(harm_categories=["category1", "category2"], authors=["author1", "author2"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1", "category2"]
    assert result[0].authors == ["author1", "author2"]


def test_get_seed_prompts_with_multiple_elements_list_filters_additional(memory: MemoryInterface):
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
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(harm_categories=["category1", "category3"], authors=["author1", "author3"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1", "category3"]
    assert result[0].authors == ["author1", "author3"]


def test_get_seed_prompts_with_substring_filters_harm_categories(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", harm_categories=["category1"], authors=["author1"], data_type="text"),
        SeedPrompt(value="prompt2", harm_categories=["category2"], authors=["author2"], data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(harm_categories=["ory1"])
    assert len(result) == 1
    assert result[0].harm_categories == ["category1"]

    result = memory.get_seed_prompts(authors=["auth"])
    assert len(result) == 2
    assert result[0].authors == ["author1"]
    assert result[1].authors == ["author2"]


def test_get_seed_prompts_with_substring_filters_groups(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", groups=["group1"], data_type="text"),
        SeedPrompt(value="prompt2", groups=["group2"], data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(groups=["oup1"])
    assert len(result) == 1
    assert result[0].groups == ["group1"]

    result = memory.get_seed_prompts(groups=["oup"])
    assert len(result) == 2
    assert result[0].groups == ["group1"]
    assert result[1].groups == ["group2"]


def test_get_seed_prompts_with_substring_filters_parameters(memory: MemoryInterface):
    seed_prompts = [
        SeedPrompt(value="prompt1", parameters=["param1"], data_type="text"),
        SeedPrompt(value="prompt2", parameters=["param2"], data_type="text"),
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts, added_by="test")

    result = memory.get_seed_prompts(parameters=["ram1"])
    assert len(result) == 1
    assert result[0].parameters == ["param1"]

    result = memory.get_seed_prompts(parameters=["ram"])
    assert len(result) == 2
    assert result[0].parameters == ["param1"]
    assert result[1].parameters == ["param2"]


def test_add_seed_prompts_to_memory_empty_list(memory: MemoryInterface):
    prompts: list[SeedPrompt] = []
    memory.add_seed_prompts_to_memory(prompts=prompts, added_by="tester")
    stored_prompts = memory.get_seed_prompts(dataset_name="test_dataset")
    assert len(stored_prompts) == 0


def test_get_seed_prompt_dataset_names_empty(memory: MemoryInterface):
    assert memory.get_seed_prompt_dataset_names() == []


def test_get_seed_prompt_dataset_names_single(memory: MemoryInterface):
    dataset_name = "test_dataset"
    seed_prompt = SeedPrompt(value="test_value", dataset_name=dataset_name, added_by="tester", data_type="text")
    memory.add_seed_prompts_to_memory(prompts=[seed_prompt])
    assert memory.get_seed_prompt_dataset_names() == [dataset_name]


def test_get_seed_prompt_dataset_names_single_dataset_multiple_entries(memory: MemoryInterface):
    dataset_name = "test_dataset"
    seed_prompt1 = SeedPrompt(value="test_value", dataset_name=dataset_name, added_by="tester", data_type="text")
    seed_prompt2 = SeedPrompt(value="test_value", dataset_name=dataset_name, added_by="tester", data_type="text")
    memory.add_seed_prompts_to_memory(prompts=[seed_prompt1, seed_prompt2])
    assert memory.get_seed_prompt_dataset_names() == [dataset_name]


def test_get_seed_prompt_dataset_names_multiple(memory: MemoryInterface):
    dataset_names = [f"dataset_{i}" for i in range(5)]
    seed_prompts = [
        SeedPrompt(value=f"value_{i}", dataset_name=dataset_name, added_by="tester", data_type="text")
        for i, dataset_name in enumerate(dataset_names)
    ]
    memory.add_seed_prompts_to_memory(prompts=seed_prompts)
    assert len(memory.get_seed_prompt_dataset_names()) == 5
    assert sorted(memory.get_seed_prompt_dataset_names()) == sorted(dataset_names)


def test_add_seed_prompt_groups_to_memory_empty_list(memory: MemoryInterface):
    prompt_group = SeedPromptGroup(
        prompts=[SeedPrompt(value="Test prompt", added_by="tester", data_type="text", sequence=0)]
    )
    prompt_group.prompts = []
    with pytest.raises(ValueError, match="Prompt group must have at least one prompt."):
        memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


def test_add_seed_prompt_groups_to_memory_single_element(memory: MemoryInterface):
    prompt = SeedPrompt(value="Test prompt", added_by="tester", data_type="text", sequence=0)
    prompt_group = SeedPromptGroup(prompts=[prompt])
    memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group], added_by="tester")
    assert len(memory.get_seed_prompts()) == 1


def test_add_seed_prompt_groups_to_memory_multiple_elements(memory: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0)
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1)
    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group], added_by="tester")
    assert len(memory.get_seed_prompts()) == 2
    assert len(memory.get_seed_prompt_groups()) == 1


def test_add_seed_prompt_groups_to_memory_no_elements(memory: MemoryInterface):
    with pytest.raises(ValueError, match="SeedPromptGroup cannot be empty."):
        prompt_group = SeedPromptGroup(prompts=[])
        memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


def test_add_seed_prompt_groups_to_memory_single_element_no_added_by(memory: MemoryInterface):
    prompt = SeedPrompt(value="Test prompt", data_type="text", sequence=0)
    prompt_group = SeedPromptGroup(prompts=[prompt])
    with pytest.raises(ValueError, match="The 'added_by' attribute must be set for each prompt."):
        memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


def test_add_seed_prompt_groups_to_memory_multiple_elements_no_added_by(memory: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", data_type="text", sequence=0)
    prompt2 = SeedPrompt(value="Test prompt 2", data_type="text", sequence=1)
    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    with pytest.raises(ValueError, match="The 'added_by' attribute must be set for each prompt."):
        memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


def test_add_seed_prompt_groups_to_memory_inconsistent_group_ids(memory: MemoryInterface):
    prompt1 = SeedPrompt(
        value="Test prompt 1", added_by="tester", prompt_group_id=uuid4(), data_type="text", sequence=0
    )
    prompt2 = SeedPrompt(
        value="Test prompt 2", added_by="tester", prompt_group_id=uuid4(), data_type="text", sequence=1
    )
    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    with pytest.raises(ValueError):
        memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])


def test_add_seed_prompt_groups_to_memory_single_element_with_added_by(memory: MemoryInterface):
    prompt = SeedPrompt(value="Test prompt", added_by="tester", data_type="text", sequence=0)
    prompt_group = SeedPromptGroup(prompts=[prompt])
    memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])
    assert len(memory.get_seed_prompts()) == 1


def test_add_seed_prompt_groups_to_memory_multiple_elements_with_added_by(memory: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0)
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1)
    prompt_group = SeedPromptGroup(prompts=[prompt1, prompt2])
    memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])
    assert len(memory.get_seed_prompts()) == 2


def test_add_seed_prompt_groups_to_memory_multiple_groups_with_added_by(memory: MemoryInterface):
    prompt1 = SeedPrompt(value="Test prompt 1", added_by="tester", data_type="text", sequence=0)
    prompt2 = SeedPrompt(value="Test prompt 2", added_by="tester", data_type="text", sequence=1)
    prompt3 = SeedPrompt(value="Test prompt 3", added_by="tester", data_type="text", sequence=0)
    prompt4 = SeedPrompt(value="Test prompt 4", added_by="tester", data_type="text", sequence=1)

    prompt_group1 = SeedPromptGroup(prompts=[prompt1, prompt2])
    prompt_group2 = SeedPromptGroup(prompts=[prompt3, prompt4])

    memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group1, prompt_group2])
    assert len(memory.get_seed_prompts()) == 4
    groups_from_memory = memory.get_seed_prompt_groups()
    assert len(groups_from_memory) == 2
    assert groups_from_memory[0].prompts[0].id != groups_from_memory[1].prompts[1].id
    assert groups_from_memory[0].prompts[0].prompt_group_id == groups_from_memory[0].prompts[1].prompt_group_id
    assert groups_from_memory[1].prompts[0].prompt_group_id == groups_from_memory[1].prompts[1].prompt_group_id


def test_get_seed_prompts_with_param_filters(memory: MemoryInterface):
    template_value = "Test template {{param1}}"
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
    memory.add_seed_prompts_to_memory(prompts=[template])

    templates = memory.get_seed_prompts(
        value=template_value,
        dataset_name=dataset_name,
        harm_categories=harm_categories,
        added_by=added_by,
        parameters=parameters,
    )
    assert len(templates) == 1
    assert templates[0].value == template_value


def test_get_seed_prompt_groups_empty(memory: MemoryInterface):
    assert memory.get_seed_prompt_groups() == []


def test_get_seed_prompt_groups_with_dataset_name(memory: MemoryInterface):
    dataset_name = "test_dataset"
    prompt_group = SeedPromptGroup(
        prompts=[
            SeedPrompt(value="Test prompt", dataset_name=dataset_name, added_by="tester", data_type="text", sequence=0)
        ]
    )
    memory.add_seed_prompt_groups_to_memory(prompt_groups=[prompt_group])

    groups = memory.get_seed_prompt_groups(dataset_name=dataset_name)
    assert len(groups) == 1
    assert groups[0].prompts[0].dataset_name == dataset_name


def test_get_seed_prompt_groups_with_multiple_filters(memory: MemoryInterface):
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
    memory.add_seed_prompt_groups_to_memory(prompt_groups=[group])

    groups = memory.get_seed_prompt_groups(
        dataset_name=dataset_name,
        data_types=data_types,
        harm_categories=harm_categories,
        added_by=added_by,
    )
    assert len(groups) == 1
    assert groups[0].prompts[0].dataset_name == dataset_name
    assert groups[0].prompts[0].added_by == added_by


def test_get_seed_prompt_groups_multiple_groups(memory: MemoryInterface):
    group1 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 1", dataset_name="dataset_1", added_by="user1", sequence=0, data_type="text")]
    )
    group2 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 2", dataset_name="dataset_2", added_by="user2", sequence=0, data_type="text")]
    )
    memory.add_seed_prompt_groups_to_memory(prompt_groups=[group1, group2])

    groups = memory.get_seed_prompt_groups()
    assert len(groups) == 2


def test_get_seed_prompt_groups_multiple_groups_with_unique_ids(memory: MemoryInterface):
    group1 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 1", dataset_name="dataset_1", added_by="user1", sequence=0, data_type="text")]
    )
    group2 = SeedPromptGroup(
        prompts=[SeedPrompt(value="Prompt 2", dataset_name="dataset_2", added_by="user2", sequence=0, data_type="text")]
    )
    memory.add_seed_prompt_groups_to_memory(prompt_groups=[group1, group2])

    groups = memory.get_seed_prompt_groups()
    assert len(groups) == 2
    # Check that each group has a unique prompt_group_id
    assert groups[0].prompts[0].prompt_group_id != groups[1].prompts[0].prompt_group_id


def test_export_all_conversations_with_scores_file_created(memory: MemoryInterface):
    memory.exporter = MemoryExporter()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        with (
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_all_prompt_pieces") as mock_get_pieces,
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_scores_by_prompt_ids") as mock_get_scores,
        ):
            file_path = Path(temp_file.name)

            mock_get_pieces.return_value = [MagicMock(original_prompt_id="1234", converted_value="sample piece")]
            mock_get_scores.return_value = [MagicMock(prompt_request_response_id="1234", score_value=10)]
            memory.export_all_conversations_with_scores(file_path=file_path)

            assert file_path.exists()


def test_export_all_conversations_with_scores_correct_data(memory: MemoryInterface):
    memory.exporter = MemoryExporter()
    expected_data = [
        {
            "prompt_request_response_id": "1234",
            "conversation": ["sample piece"],
            "score_value": 10,
        }
    ]

    with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as temp_file:
        with (
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_all_prompt_pieces") as mock_get_pieces,
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_scores_by_prompt_ids") as mock_get_scores,
            patch.object(memory.exporter, "export_data") as mock_export_data,
        ):
            file_path = Path(temp_file.name)

            mock_get_pieces.return_value = [MagicMock(original_prompt_id="1234", converted_value="sample piece")]
            mock_get_scores.return_value = [MagicMock(prompt_request_response_id="1234", score_value=10)]

            memory.export_all_conversations_with_scores(file_path=file_path)

            mock_export_data.assert_called_once_with(expected_data, file_path=file_path, export_type="json")
            assert mock_export_data.call_args[0][0] == expected_data


def test_export_all_conversations_with_scores_empty_data(memory: MemoryInterface):
    memory.exporter = MemoryExporter()
    expected_data: list = []
    with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as temp_file:
        with (
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_all_prompt_pieces") as mock_get_pieces,
            patch("pyrit.memory.duckdb_memory.DuckDBMemory.get_scores_by_prompt_ids") as mock_get_scores,
            patch.object(memory.exporter, "export_data") as mock_export_data,
        ):
            file_path = Path(temp_file.name)

            mock_get_pieces.return_value = []
            mock_get_scores.return_value = []

            memory.export_all_conversations_with_scores(file_path=file_path)
            mock_export_data.assert_called_once_with(expected_data, file_path=file_path, export_type="json")
