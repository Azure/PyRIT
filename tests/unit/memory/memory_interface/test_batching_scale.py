# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for batching functionality to handle large numbers of IDs.
This addresses the scaling bug where methods like get_scores_by_prompt_ids
fail when querying with many IDs due to SQLite bind variable limits.
"""

import uuid

from pyrit.memory import MemoryInterface
from pyrit.memory.memory_interface import _SQLITE_MAX_BIND_VARS
from pyrit.models import MessagePiece, Score


def _create_message_piece(conversation_id: str = None, original_value: str = "test message") -> MessagePiece:
    """Create a sample message piece for testing."""
    return MessagePiece(
        id=str(uuid.uuid4()),
        role="user",
        original_value=original_value,
        converted_value=original_value,
        sequence=0,
        conversation_id=conversation_id or str(uuid.uuid4()),
        labels={"test": "label"},
        attack_identifier={"id": str(uuid.uuid4())},
    )


def _create_score(message_piece_id: str) -> Score:
    """Create a sample score for testing."""
    return Score(
        score_value="0.5",
        score_value_description="test score",
        score_type="float_scale",
        score_category=["test"],
        score_rationale="test rationale",
        score_metadata={},
        scorer_class_identifier={"__type__": "TestScorer"},
        message_piece_id=message_piece_id,
    )


class TestBatchingScale:
    """Tests for batching when querying with many IDs."""

    def test_get_message_pieces_with_many_prompt_ids(self, sqlite_instance: MemoryInterface):
        """Test that get_message_pieces works with more IDs than the batch limit."""
        # Create more message pieces than the batch limit
        num_pieces = _SQLITE_MAX_BIND_VARS + 100
        pieces = [_create_message_piece() for _ in range(num_pieces)]

        # Add to memory
        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        # Query with all IDs - this should work with batching
        all_ids = [piece.id for piece in pieces]
        results = sqlite_instance.get_message_pieces(prompt_ids=all_ids)

        assert len(results) == num_pieces, f"Expected {num_pieces} results, got {len(results)}"

    def test_get_message_pieces_with_exact_batch_size(self, sqlite_instance: MemoryInterface):
        """Test that get_message_pieces works with exactly the batch limit."""
        num_pieces = _SQLITE_MAX_BIND_VARS
        pieces = [_create_message_piece() for _ in range(num_pieces)]

        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        all_ids = [piece.id for piece in pieces]
        results = sqlite_instance.get_message_pieces(prompt_ids=all_ids)

        assert len(results) == num_pieces

    def test_get_message_pieces_with_double_batch_size(self, sqlite_instance: MemoryInterface):
        """Test that get_message_pieces works with double the batch limit."""
        num_pieces = _SQLITE_MAX_BIND_VARS * 2
        pieces = [_create_message_piece() for _ in range(num_pieces)]

        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        all_ids = [piece.id for piece in pieces]
        results = sqlite_instance.get_message_pieces(prompt_ids=all_ids)

        assert len(results) == num_pieces

    def test_get_scores_with_many_score_ids(self, sqlite_instance: MemoryInterface):
        """Test that get_scores works with more IDs than the batch limit."""
        # Create message pieces first (scores need to reference them)
        num_scores = _SQLITE_MAX_BIND_VARS + 100
        pieces = [_create_message_piece() for _ in range(num_scores)]
        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        # Create and add scores
        scores = [_create_score(str(piece.id)) for piece in pieces]
        sqlite_instance.add_scores_to_memory(scores=scores)

        # Query with all score IDs - this should work with batching
        all_score_ids = [str(score.id) for score in scores]
        results = sqlite_instance.get_scores(score_ids=all_score_ids)

        assert len(results) == num_scores, f"Expected {num_scores} results, got {len(results)}"

    def test_get_prompt_scores_with_many_prompt_ids(self, sqlite_instance: MemoryInterface):
        """Test that get_prompt_scores works with more prompt IDs than the batch limit."""
        # Create message pieces
        num_pieces = _SQLITE_MAX_BIND_VARS + 50
        pieces = [_create_message_piece() for _ in range(num_pieces)]
        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        # Create and add scores for half of them
        num_scores = num_pieces // 2
        scores = [_create_score(str(pieces[i].id)) for i in range(num_scores)]
        sqlite_instance.add_scores_to_memory(scores=scores)

        # Query with all prompt IDs - should return scores for pieces that have them
        all_prompt_ids = [piece.id for piece in pieces]
        results = sqlite_instance.get_prompt_scores(prompt_ids=all_prompt_ids)

        assert len(results) == num_scores, f"Expected {num_scores} results, got {len(results)}"

    def test_get_message_pieces_batching_preserves_other_filters(self, sqlite_instance: MemoryInterface):
        """Test that batching still applies other filter conditions correctly."""
        # Create pieces with different roles
        num_pieces = _SQLITE_MAX_BIND_VARS + 50
        user_pieces = [_create_message_piece() for _ in range(num_pieces)]
        for piece in user_pieces:
            piece.role = "user"

        assistant_pieces = [_create_message_piece() for _ in range(50)]
        for piece in assistant_pieces:
            piece.role = "assistant"

        all_pieces = user_pieces + assistant_pieces
        sqlite_instance.add_message_pieces_to_memory(message_pieces=all_pieces)

        # Query with all IDs but filter by role
        all_ids = [piece.id for piece in all_pieces]
        results = sqlite_instance.get_message_pieces(prompt_ids=all_ids, role="user")

        assert len(results) == num_pieces, f"Expected {num_pieces} user pieces, got {len(results)}"

    def test_get_message_pieces_small_list_still_works(self, sqlite_instance: MemoryInterface):
        """Test that small ID lists (under batch limit) still work correctly."""
        num_pieces = 10
        pieces = [_create_message_piece() for _ in range(num_pieces)]
        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        all_ids = [piece.id for piece in pieces]
        results = sqlite_instance.get_message_pieces(prompt_ids=all_ids)

        assert len(results) == num_pieces

    def test_get_message_pieces_with_many_original_values(self, sqlite_instance: MemoryInterface):
        """Test that get_message_pieces works with many original_values exceeding batch limit."""
        num_pieces = _SQLITE_MAX_BIND_VARS + 100
        # Create pieces with unique original values
        pieces = [_create_message_piece(original_value=f"unique_value_{i}") for i in range(num_pieces)]
        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        # Query with all original values
        all_values = [piece.original_value for piece in pieces]
        results = sqlite_instance.get_message_pieces(original_values=all_values)

        assert len(results) == num_pieces, f"Expected {num_pieces} results, got {len(results)}"

    def test_get_message_pieces_with_many_converted_value_sha256(self, sqlite_instance: MemoryInterface):
        """Test that get_message_pieces works with many converted_value_sha256 exceeding batch limit."""
        num_pieces = _SQLITE_MAX_BIND_VARS + 100
        pieces = [_create_message_piece(original_value=f"unique_value_{i}") for i in range(num_pieces)]
        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        # Get SHA256 hashes from stored pieces
        stored_pieces = sqlite_instance.get_message_pieces()
        all_hashes = [piece.converted_value_sha256 for piece in stored_pieces if piece.converted_value_sha256]

        if len(all_hashes) > _SQLITE_MAX_BIND_VARS:
            results = sqlite_instance.get_message_pieces(converted_value_sha256=all_hashes)
            assert len(results) == len(all_hashes)
