# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
from pathlib import Path

import pytest

from pyrit.models import PromptResponse
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse


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
        index=1,
        finish_reason="test",
        api_request_time_to_complete_ns=1,
    )


@pytest.fixture
def prompt_request_pieces() -> list[PromptRequestPiece]:
    return [
        PromptRequestPiece(
            role="user",
            original_value="First piece",
            conversation_id="test-conversation-1",
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="Second piece",
            conversation_id="test-conversation-1",
        ),
        PromptRequestPiece(
            role="user",
            original_value="Third piece",
            conversation_id="test-conversation-1",
        ),
    ]


@pytest.fixture
def prompt_request_response(prompt_request_pieces) -> PromptRequestResponse:
    return PromptRequestResponse(request_pieces=prompt_request_pieces)


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


def test_get_piece_returns_correct_piece(prompt_request_response: PromptRequestResponse) -> None:
    # Test getting first piece (default)
    first_piece = prompt_request_response.get_piece()
    assert first_piece.original_value == "First piece"
    assert first_piece.role == "user"

    # Test getting specific pieces by index
    second_piece = prompt_request_response.get_piece(1)
    assert second_piece.original_value == "Second piece"
    assert second_piece.role == "assistant"

    third_piece = prompt_request_response.get_piece(2)
    assert third_piece.original_value == "Third piece"
    assert third_piece.role == "user"


def test_get_piece_raises_index_error_for_invalid_index(prompt_request_response: PromptRequestResponse) -> None:
    with pytest.raises(IndexError, match="No request piece at index 3"):
        prompt_request_response.get_piece(3)


def test_get_piece_raises_value_error_for_empty_request() -> None:
    empty_response = PromptRequestResponse(request_pieces=[])
    with pytest.raises(ValueError, match="Empty request pieces"):
        empty_response.get_piece()


def test_filter_by_role_returns_correct_pieces(prompt_request_response: PromptRequestResponse) -> None:
    # Filter by user role
    user_pieces = prompt_request_response.filter_by_role(role="user")
    assert len(user_pieces) == 2
    assert all(piece.role == "user" for piece in user_pieces)
    assert user_pieces[0].original_value == "First piece"
    assert user_pieces[1].original_value == "Third piece"

    # Filter by assistant role
    assistant_pieces = prompt_request_response.filter_by_role(role="assistant")
    assert len(assistant_pieces) == 1
    assert assistant_pieces[0].role == "assistant"
    assert assistant_pieces[0].original_value == "Second piece"


def test_filter_by_role_returns_empty_for_nonexistent_role(prompt_request_response: PromptRequestResponse) -> None:
    system_pieces = prompt_request_response.filter_by_role(role="system")
    assert len(system_pieces) == 0
    assert isinstance(system_pieces, list)


def test_filter_by_role_with_empty_request() -> None:
    empty_response = PromptRequestResponse(request_pieces=[])
    filtered_pieces = empty_response.filter_by_role(role="user")
    assert len(filtered_pieces) == 0
    assert isinstance(filtered_pieces, list)
