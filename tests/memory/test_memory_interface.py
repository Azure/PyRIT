# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
from unittest.mock import MagicMock, patch
import pytest
import random
from string import ascii_lowercase

from pyrit.memory import MemoryInterface
from pyrit.memory.memory_models import PromptRequestPiece
from pyrit.models import PromptRequestResponse

from tests.mocks import get_memory_interface, get_sample_conversations


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


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


def test_add_request_pieces_to_memory_calls_validate(memory: MemoryInterface):
    request_response = MagicMock(PromptRequestResponse)
    request_response.request_pieces = [MagicMock(PromptRequestPiece)]
    with (
        patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_request_pieces_to_memory"),
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

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_request_pieces_to_memory") as mock_add:
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

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_request_pieces_to_memory") as mock_add:
        memory.add_request_response_to_memory(request=PromptRequestResponse(request_pieces=sample_conversations))
        assert mock_add.called

        args, kwargs = mock_add.call_args
        assert kwargs["request_pieces"][0].sequence == 1, "Sequence should increment previous conversation by 1"
        assert kwargs["request_pieces"][1].sequence == 1
        assert kwargs["request_pieces"][2].sequence == 1


def test_add_response_entries_to_memory_updates_sequence(
    memory: MemoryInterface, sample_conversations: list[PromptRequestPiece]
):
    request = [sample_conversations[0]]
    memory.add_request_response_to_memory(request=PromptRequestResponse(request_pieces=request))

    with patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_request_pieces_to_memory") as mock_add:
        memory.add_response_entries_to_memory(request=request[0], response_text_pieces=["test"])

        assert mock_add.called

        args, kwargs = mock_add.call_args
        assert kwargs["request_pieces"][0].sequence == 1, "Sequence should increment previous conversation by 1"


def test_insert_prompt_memories_inserts_embedding(
    memory: MemoryInterface, sample_conversations: list[PromptRequestPiece]
):

    request = PromptRequestResponse(request_pieces=[sample_conversations[0]])

    embedding_mock = MagicMock()
    embedding_mock.generate_text_embedding.returns = [0, 1, 2]
    memory.enable_embedding(embedding_model=embedding_mock)

    with (
        patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_request_pieces_to_memory"),
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
        patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_request_pieces_to_memory"),
        patch("pyrit.memory.duckdb_memory.DuckDBMemory._add_embeddings_to_memory") as mock_embedding,
    ):

        memory.add_request_response_to_memory(request=request)

        assert mock_embedding.assert_not_called


def test_get_orchestrator_conversation_sorting(memory: MemoryInterface, sample_conversations: list[PromptRequestPiece]):
    conversation_id = sample_conversations[0].orchestrator_identifier["id"]

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
        orchestrator_id = int(sample_conversations[0].orchestrator_identifier["id"])

        response = memory.get_orchestrator_conversations(orchestrator_id=orchestrator_id)

        current_value = response[0].conversation_id
        for obj in response[1:]:
            new_value = obj.conversation_id
            if new_value != current_value:
                if any(o.conversation_id == current_value for o in response[response.index(obj) :]):
                    assert False, "Conversation IDs are not grouped together"
