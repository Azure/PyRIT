# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
import random
from string import ascii_lowercase

from pyrit.common.path import RESULTS_PATH
from pyrit.memory import MemoryInterface
from pyrit.memory.memory_exporter import MemoryExporter
from pyrit.memory.memory_models import PromptRequestPiece, PromptMemoryEntry
from pyrit.models import PromptRequestResponse
from pyrit.orchestrator import Orchestrator
from pyrit.score import Score

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
