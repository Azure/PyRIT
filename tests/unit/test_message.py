# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
from pathlib import Path

import pytest

from pyrit.models import PromptResponse
from pyrit.models import (
    MessagePiece,
    Message,
)


@pytest.fixture
def prompt_response_1() -> PromptResponse:
    return PromptResponse(
        completion="This is a test",
        prompt="This is a test",
        id="1234",
        completion_tokens=1,
        prompt_tokens=1,
        total_tokens=1,
        model="test",
        object="test",
        created_at=1,
        logprobs=True,
    )


@pytest.fixture
def message_pieces() -> list[MessagePiece]:
    return [
        MessagePiece(
            role="user",
            original_value="First piece",
            conversation_id="test-conversation-1",
        ),
        MessagePiece(
            role="user",
            original_value="Second piece",
            conversation_id="test-conversation-1",
        ),
        MessagePiece(
            role="user",
            original_value="Third piece",
            conversation_id="test-conversation-1",
        ),
    ]


@pytest.fixture
def message(message_pieces) -> Message:
    return Message(message_pieces=message_pieces)


def test_saving_of_prompt_response(prompt_response_1: PromptResponse) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        new_file = prompt_response_1.save_to_file(directory_path=Path(tmp_dir))
        assert new_file


def test_save_and_load_of_prompt_response(prompt_response_1: PromptResponse) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save file
        new_file = prompt_response_1.save_to_file(directory_path=Path(tmp_dir))

        # Load file
        loaded_prompt_response = PromptResponse.load_from_file(file_path=Path(new_file))
        assert loaded_prompt_response == prompt_response_1


def test_get_piece_returns_correct_piece(message: Message) -> None:
    # Test getting first piece (default)
    first_piece = message.get_piece()
    assert first_piece.original_value == "First piece"
    assert first_piece.role == "user"

    # Test getting specific pieces by index
    second_piece = message.get_piece(1)
    assert second_piece.original_value == "Second piece"
    assert second_piece.role == "user"

    third_piece = message.get_piece(2)
    assert third_piece.original_value == "Third piece"
    assert third_piece.role == "user"


def test_get_piece_raises_index_error_for_invalid_index(message: Message) -> None:
    with pytest.raises(IndexError, match="No request piece at index 3"):
        message.get_piece(3)


def test_get_piece_raises_value_error_for_empty_request() -> None:
    with pytest.raises(ValueError, match="at least one request piece"):
        Message(message_pieces=[])


def test_get_all_values_returns_all_converted_strings(message_pieces: list[MessagePiece]) -> None:
    response_one = Message(message_pieces=message_pieces[:2])
    response_two = Message(message_pieces=message_pieces[2:])

    flattened = Message.get_all_values([response_one, response_two])

    assert flattened == ["First piece", "Second piece", "Third piece"]
