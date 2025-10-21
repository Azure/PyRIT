# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.models import (
    Message,
    MessagePiece,
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
    with pytest.raises(IndexError, match="No message piece at index 3"):
        message.get_piece(3)


def test_get_piece_raises_value_error_for_empty_request() -> None:
    with pytest.raises(ValueError, match="at least one message piece"):
        Message(message_pieces=[])


def test_get_all_values_returns_all_converted_strings(message_pieces: list[MessagePiece]) -> None:
    response_one = Message(message_pieces=message_pieces[:2])
    response_two = Message(message_pieces=message_pieces[2:])

    flattened = Message.get_all_values([response_one, response_two])

    assert flattened == ["First piece", "Second piece", "Third piece"]
