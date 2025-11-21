# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pytest
from typing import Optional
from unittest.mock import MagicMock

from pyrit.exceptions import EmptyResponseException, PyritException
from pyrit.models import MessagePiece, PromptDataType, PromptResponseError
from pyrit.prompt_target.openai.openai_response_target import MessagePieceType
from pyrit.prompt_target.openai.openai_error_handling import construct_response_message


def create_mock_response_from_dict(response_dict: dict) -> MagicMock:
    """
    Helper to create a mock OpenAI Response object from a dict.
    Converts JSON test fixtures to mock SDK Response objects.
    """
    mock_response = MagicMock()
    
    # Set error attribute
    error_dict = response_dict.get("error")
    if error_dict:
        mock_error = MagicMock()
        mock_error.code = error_dict.get("code", "")
        mock_error.message = error_dict.get("message", "")
        mock_response.error = mock_error
    else:
        mock_response.error = None
    
    # Set status
    mock_response.status = response_dict.get("status")
    
    # Set output sections as list of dicts (sections can be accessed directly)
    mock_response.output = response_dict.get("output")
    
    return mock_response


def simple_parse_section(*, section: dict, message_piece: MessagePiece, error: Optional[PromptResponseError]) -> MessagePiece | None:
    """
    Simple test helper to parse output sections without needing the full target logic.
    """
    section_type = section.get("type", "")
    piece_value = ""
    piece_type: PromptDataType = "text"
    
    if section_type == MessagePieceType.MESSAGE:
        section_content = section.get("content", [])
        if len(section_content) == 0:
            raise EmptyResponseException(message="The chat returned an empty message section.")
        # Handle both "text" and "output_text" keys
        piece_value = section_content[0].get("text") or section_content[0].get("output_text", "")
    elif section_type in (MessagePieceType.REASONING, MessagePieceType.FUNCTION_CALL):
        piece_value = json.dumps(section, separators=(",", ":"))
        piece_type = section_type
    elif section_type == MessagePieceType.WEB_SEARCH_CALL:
        piece_value = json.dumps(section, separators=(",", ":"))
        piece_type = "tool_call"  # web_search_call is represented as tool_call data type
    elif section_type == "custom_tool_call":
        piece_value = section.get("input", "")
        if len(piece_value) == 0:
            raise EmptyResponseException(message="The chat returned an empty message section.")
    elif section_type == "image_generation_call":
        # Return None to skip this section
        return None
    else:
        raise ValueError(f"Unknown section type: {section_type}")
    
    if len(piece_value) == 0:
        raise EmptyResponseException(message="The chat returned an empty response.")
    
    return MessagePiece(
        role=message_piece.role,
        original_value=piece_value,
        converted_value=piece_value,
        original_value_data_type=piece_type,
        converted_value_data_type=piece_type,
        converter_identifiers=message_piece.converter_identifiers,
    )


@pytest.fixture
def dummy_text_message_piece() -> MessagePiece:
    return MessagePiece(
        role="user",
        original_value="original prompt text",
        converted_value="Hello, how are you?",
        conversation_id="12345678-1234-5678-1234-567812345678",
        sequence=0,
    )


@pytest.mark.parametrize(
    "status",
    ["failed", "in_progress", "cancelled", None],
)
def test_construct_message_not_completed_status(
    status: str, dummy_text_message_piece: MessagePiece
):
    response_dict = {"status": f"{status}", "error": {"code": "some_error_code", "message": "An error occurred"}}
    mock_response = create_mock_response_from_dict(response_dict)

    with pytest.raises(PyritException) as excinfo:
        construct_response_message(
            completion_response=mock_response,
            message_piece=dummy_text_message_piece,
            parse_section_fn=simple_parse_section
        )
    assert f"Response error: some_error_code" in str(excinfo.value)


def test_construct_message_empty_response(dummy_text_message_piece: MessagePiece):
    response_dict = {
        "status": "completed",
        "output": [{"type": "message", "content": [{"text": ""}]}],
    }
    mock_response = create_mock_response_from_dict(response_dict)

    with pytest.raises(EmptyResponseException) as excinfo:
        construct_response_message(
            completion_response=mock_response,
            message_piece=dummy_text_message_piece,
            parse_section_fn=simple_parse_section
        )
    assert "The chat returned an empty response." in str(excinfo.value)


def test_construct_message_invalid_status(dummy_text_message_piece: MessagePiece):
    # Should raise PyritException for invalid status
    response_dict = {"status": "failed", "error": None, "output": []}
    mock_response = create_mock_response_from_dict(response_dict)
    
    with pytest.raises(PyritException) as excinfo:
        construct_response_message(
            completion_response=mock_response,
            message_piece=dummy_text_message_piece,
            parse_section_fn=simple_parse_section
        )
    assert "Unexpected status: failed" in str(excinfo.value)


def test_construct_message_none_status(dummy_text_message_piece: MessagePiece):
    # Should raise PyritException for missing status (None)
    response_dict = {"status": None, "error": None, "output": [{"type": "message", "content": [{"text": "hi"}]}]}
    mock_response = create_mock_response_from_dict(response_dict)
    
    with pytest.raises(PyritException) as excinfo:
        construct_response_message(
            completion_response=mock_response,
            message_piece=dummy_text_message_piece,
            parse_section_fn=simple_parse_section
        )
    assert "Unexpected status: None" in str(excinfo.value)


def test_construct_message_reasoning(dummy_text_message_piece: MessagePiece):
    # Should handle reasoning type and skip empty summaries
    reasoning_json = {
        "status": "completed",
        "output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "Reasoning summary."}]}],
    }
    mock_response = create_mock_response_from_dict(reasoning_json)
    response = construct_response_message(
        completion_response=mock_response,
        message_piece=dummy_text_message_piece,
        parse_section_fn=simple_parse_section
    )
    piece = response.message_pieces[0]
    assert piece.original_value_data_type == "reasoning"
    section = json.loads(piece.original_value)
    assert section["type"] == "reasoning"
    assert section["summary"][0]["text"] == "Reasoning summary."


def test_construct_message_function_call(dummy_text_message_piece: MessagePiece):
    func_call_json = {
        "status": "completed",
        "output": [{"type": "function_call", "name": "do_something", "arguments": '{"x":1}'}],
    }
    mock_response = create_mock_response_from_dict(func_call_json)
    resp = construct_response_message(
        completion_response=mock_response,
        message_piece=dummy_text_message_piece,
        parse_section_fn=simple_parse_section
    )
    piece = resp.message_pieces[0]
    assert piece.original_value_data_type == "function_call"
    section = json.loads(piece.original_value)
    assert section["type"] == "function_call"
    assert section["name"] == "do_something"


def test_construct_message_forwards_web_search_call(dummy_text_message_piece: MessagePiece):
    body = {
        "status": "completed",
        "output": [{"type": "web_search_call", "query": "time in Tokyo", "provider": "bing"}],
    }
    mock_response = create_mock_response_from_dict(body)
    resp = construct_response_message(
        completion_response=mock_response,
        message_piece=dummy_text_message_piece,
        parse_section_fn=simple_parse_section
    )
    assert len(resp.message_pieces) == 1
    p = resp.message_pieces[0]
    assert p.original_value_data_type == "tool_call"
    section = json.loads(p.original_value)
    assert section["type"] == "web_search_call"
    assert section["query"] == "time in Tokyo"


def test_construct_message_skips_unhandled_types(dummy_text_message_piece: MessagePiece):
    body = {
        "status": "completed",
        "output": [
            {"type": "image_generation_call", "prompt": "cat astronaut"},  # currently unhandled -> skipped
            {"type": "message", "content": [{"type": "output_text", "text": "Hi"}]},
        ],
    }
    mock_response = create_mock_response_from_dict(body)
    resp = construct_response_message(
        completion_response=mock_response,
        message_piece=dummy_text_message_piece,
        parse_section_fn=simple_parse_section
    )
    # Only the 'message' section becomes a piece; image_generation_call is skipped
    assert len(resp.message_pieces) == 1
    assert resp.message_pieces[0].original_value == "Hi"


def test_construct_message_content_filter_error(dummy_text_message_piece: MessagePiece):
    # Should handle content_filter error code
    response_dict = {
        "status": "completed",
        "error": {"code": "content_filter", "message": "Content was filtered"},
        "output": []
    }
    mock_response = create_mock_response_from_dict(response_dict)
    # Add model_dump_json for error handling
    mock_response.model_dump_json.return_value = json.dumps(response_dict)
    
    # This should return an error message, not raise
    result = construct_response_message(
        completion_response=mock_response,
        message_piece=dummy_text_message_piece,
        parse_section_fn=simple_parse_section
    )
    
    # Check that it returns an error response (Message object with error state)
    assert result is not None
    # The error is in the message pieces
    assert len(result.message_pieces) == 1
    assert result.message_pieces[0].response_error == "blocked"


# Tests for Chat Completions API (construct_chat_completion_message)

from pyrit.prompt_target.openai.openai_error_handling import construct_chat_completion_message


def create_mock_chat_completion(content: str = "hi", finish_reason: str = "stop") -> MagicMock:
    """Helper to create a mock OpenAI ChatCompletion response"""
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].finish_reason = finish_reason
    mock_completion.choices[0].message.content = content
    mock_completion.model_dump_json.return_value = json.dumps({
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}]
    })
    return mock_completion


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_chat_completion_valid_finish_reasons(
    finish_reason: str, dummy_text_message_piece: MessagePiece
):
    mock_completion = create_mock_chat_completion(content="Hello from chat", finish_reason=finish_reason)

    result = construct_chat_completion_message(
        completion_response=mock_completion, message_piece=dummy_text_message_piece
    )

    assert len(result.message_pieces) == 1
    assert result.get_value() == "Hello from chat"


def test_chat_completion_empty_response(dummy_text_message_piece: MessagePiece):
    mock_completion = create_mock_chat_completion(content="", finish_reason="stop")

    with pytest.raises(EmptyResponseException) as excinfo:
        construct_chat_completion_message(
            completion_response=mock_completion, message_piece=dummy_text_message_piece
        )
    assert "The chat returned an empty response." in str(excinfo.value)


def test_chat_completion_unknown_finish_reason(dummy_text_message_piece: MessagePiece):
    mock_completion = create_mock_chat_completion(content="Some content", finish_reason="unexpected")

    with pytest.raises(PyritException) as excinfo:
        construct_chat_completion_message(
            completion_response=mock_completion, message_piece=dummy_text_message_piece
        )
    assert "Unknown finish_reason" in str(excinfo.value)


def test_chat_completion_content_filter(dummy_text_message_piece: MessagePiece):
    mock_completion = create_mock_chat_completion(content="", finish_reason="content_filter")

    # Should return an error message, not raise
    result = construct_chat_completion_message(
        completion_response=mock_completion, message_piece=dummy_text_message_piece
    )
    
    # Check that it returns an error response
    assert result is not None
    assert len(result.message_pieces) == 1
    assert result.message_pieces[0].response_error == "blocked"

