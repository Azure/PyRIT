# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid

import pytest

from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import MessagePiece, Message
from pyrit.score.true_false.decoding_scorer import DecodingScorer


def create_conversation(
    user_original: str,
    assistant_response: str,
    memory: MemoryInterface,
    user_converted: str = "encoded",
    user_metadata: dict | None = None,
) -> Message:
    """Helper to create a conversation and return the assistant response."""
    conversation_id = str(uuid.uuid4())
    
    user_request = Message(
        message_pieces=[
            MessagePiece(
                role="user",
                original_value=user_original,
                converted_value=user_converted,
                original_value_data_type="text",
                converted_value_data_type="text",
                conversation_id=conversation_id,
                prompt_metadata=user_metadata or {},
            )
        ]
    )
    memory.add_message_to_memory(request=user_request)
    
    assistant = Message(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value=assistant_response,
                converted_value=assistant_response,
                original_value_data_type="text",
                converted_value_data_type="text",
                conversation_id=conversation_id,
            )
        ]
    )
    memory.add_message_to_memory(request=assistant)
    
    return assistant


@pytest.mark.asyncio
async def test_decoding_scorer_init_defaults(patch_central_database):
    """Test initialization with default parameters."""
    scorer = DecodingScorer()
    assert scorer._case_insensitive is True
    assert scorer._score_categories == []
    assert scorer._request_text_type == "original_value"


@pytest.mark.asyncio
async def test_decoding_scorer_init_custom(patch_central_database):
    """Test initialization with custom parameters."""
    scorer = DecodingScorer(
        case_insensitive=False,
        categories=["test"],
        request_text_type="metadata",
    )
    assert scorer._case_insensitive is False
    assert scorer._score_categories == ["test"]
    assert scorer._request_text_type == "metadata"


@pytest.mark.asyncio
async def test_decoding_scorer_case_insensitive_match(sqlite_instance: MemoryInterface):
    """Test case-insensitive matching."""
    assistant = create_conversation(
        user_original="SECRET",
        assistant_response="The secret is here",
        memory=sqlite_instance,
    )
    
    scorer = DecodingScorer(case_insensitive=True)
    scores = await scorer.score_async(assistant)
    
    assert len(scores) == 1
    assert scores[0].get_value() is True


@pytest.mark.asyncio
async def test_decoding_scorer_case_sensitive_match(sqlite_instance: MemoryInterface):
    """Test case-sensitive matching."""
    assistant = create_conversation(
        user_original="SECRET",
        assistant_response="Found SECRET here",
        memory=sqlite_instance,
    )
    
    scorer = DecodingScorer(case_insensitive=False)
    scores = await scorer.score_async(assistant)
    
    assert len(scores) == 1
    assert scores[0].get_value() is True


@pytest.mark.asyncio
async def test_decoding_scorer_case_sensitive_no_match(sqlite_instance: MemoryInterface):
    """Test case-sensitive fails when case differs."""
    assistant = create_conversation(
        user_original="SECRET",
        assistant_response="Found secret here",  # lowercase
        memory=sqlite_instance,
    )
    
    scorer = DecodingScorer(case_insensitive=False)
    scores = await scorer.score_async(assistant)
    
    assert len(scores) == 1
    assert scores[0].get_value() is False


@pytest.mark.asyncio
async def test_decoding_scorer_no_match(sqlite_instance: MemoryInterface):
    """Test when text is not found."""
    assistant = create_conversation(
        user_original="password",
        assistant_response="No match here",
        memory=sqlite_instance,
    )
    
    scorer = DecodingScorer()
    scores = await scorer.score_async(assistant)
    
    assert len(scores) == 1
    assert scores[0].get_value() is False


@pytest.mark.asyncio
async def test_decoding_scorer_request_text_type_converted(sqlite_instance: MemoryInterface):
    """Test using converted_value instead of original_value."""
    assistant = create_conversation(
        user_original="original_text",
        user_converted="DECODED_SECRET",
        assistant_response="Found DECODED_SECRET",
        memory=sqlite_instance,
    )
    
    scorer = DecodingScorer(request_text_type="converted_value")
    scores = await scorer.score_async(assistant)
    
    assert len(scores) == 1
    assert scores[0].get_value() is True


@pytest.mark.asyncio
async def test_decoding_scorer_request_text_type_metadata(sqlite_instance: MemoryInterface):
    """Test using metadata decoded_text."""
    assistant = create_conversation(
        user_original="original",
        assistant_response="Found METADATA_SECRET",
        user_metadata={"decoded_text": "METADATA_SECRET"},
        memory=sqlite_instance,
    )
    
    scorer = DecodingScorer(request_text_type="metadata")
    scores = await scorer.score_async(assistant)
    
    assert len(scores) == 1
    assert scores[0].get_value() is True


@pytest.mark.asyncio
async def test_decoding_scorer_metadata_missing(sqlite_instance: MemoryInterface):
    """Test metadata mode with missing decoded_text."""
    assistant = create_conversation(
        user_original="original",
        assistant_response="Some text",
        user_metadata={"other_key": "value"},
        memory=sqlite_instance,
    )
    
    scorer = DecodingScorer(request_text_type="metadata")
    scores = await scorer.score_async(assistant)
    
    assert len(scores) == 1
    # Empty string is in any string
    assert scores[0].get_value() is True


@pytest.mark.asyncio
async def test_decoding_scorer_with_categories(sqlite_instance: MemoryInterface):
    """Test categories are assigned."""
    assistant = create_conversation(
        user_original="test",
        assistant_response="test",
        memory=sqlite_instance,
    )
    
    scorer = DecodingScorer(categories=["cat1", "cat2"])
    scores = await scorer.score_async(assistant)
    
    assert scores[0].score_category == ["cat1", "cat2"]


@pytest.mark.asyncio
async def test_decoding_scorer_identifier(sqlite_instance: MemoryInterface):
    """Test scorer identifier is set correctly."""
    assistant = create_conversation(
        user_original="test",
        assistant_response="test",
        memory=sqlite_instance,
    )
    
    scorer = DecodingScorer()
    scores = await scorer.score_async(assistant)
    
    assert scores[0].scorer_class_identifier["__type__"] == "DecodingScorer"
