# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for batching functionality to handle large numbers of IDs.
This addresses the scaling bug where methods like get_scores_by_prompt_ids
fail when querying with many IDs due to SQLite bind variable limits.
"""

import hashlib
import uuid
from unittest.mock import MagicMock, patch

from sqlalchemy import Column, Integer
from sqlalchemy.orm import declarative_base

from pyrit.memory import MemoryInterface
from pyrit.memory.memory_interface import _SQLITE_MAX_BIND_VARS, _batched_in_condition
from pyrit.models import MessagePiece, Score


def _create_message_piece(conversation_id: str = None, original_value: str = "test message") -> MessagePiece:
    """Create a sample message piece for testing."""
    converted_value = original_value
    # Compute SHA256 for converted_value so filtering by sha256 works
    sha256 = hashlib.sha256(converted_value.encode("utf-8")).hexdigest()
    return MessagePiece(
        id=str(uuid.uuid4()),
        role="user",
        original_value=original_value,
        converted_value=converted_value,
        converted_value_sha256=sha256,
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


class TestBatchedInCondition:
    """Tests for the _batched_in_condition helper function."""

    def test_batched_in_condition_small_list(self):
        """Test that small lists generate a simple IN condition."""
        Base = declarative_base()

        class TestModel(Base):
            __tablename__ = "test"
            id = Column(Integer, primary_key=True)

        values = list(range(10))
        condition = _batched_in_condition(TestModel.id, values)

        # Should be a simple IN clause, not an OR
        assert "IN" in str(condition)
        assert "OR" not in str(condition)

    def test_batched_in_condition_exact_batch_size(self):
        """Test with exactly _SQLITE_MAX_BIND_VARS values."""
        Base = declarative_base()

        class TestModel(Base):
            __tablename__ = "test"
            id = Column(Integer, primary_key=True)

        values = list(range(_SQLITE_MAX_BIND_VARS))
        condition = _batched_in_condition(TestModel.id, values)

        # Should still be a simple IN clause at the limit
        assert "IN" in str(condition)
        # May or may not have OR depending on implementation at boundary

    def test_batched_in_condition_over_batch_size(self):
        """Test with values exceeding _SQLITE_MAX_BIND_VARS."""
        Base = declarative_base()

        class TestModel(Base):
            __tablename__ = "test"
            id = Column(Integer, primary_key=True)

        values = list(range(_SQLITE_MAX_BIND_VARS + 100))
        condition = _batched_in_condition(TestModel.id, values)

        # Should generate OR of multiple IN clauses
        condition_str = str(condition)
        assert "OR" in condition_str
        assert "IN" in condition_str

    def test_batched_in_condition_double_batch_size(self):
        """Test with double the batch size."""
        Base = declarative_base()

        class TestModel(Base):
            __tablename__ = "test"
            id = Column(Integer, primary_key=True)

        values = list(range(_SQLITE_MAX_BIND_VARS * 2))
        condition = _batched_in_condition(TestModel.id, values)

        # Should generate multiple batches
        condition_str = str(condition)
        assert "OR" in condition_str
        # Should have at least 2 IN clauses
        assert condition_str.count("IN") >= 2

    def test_batched_in_condition_three_batches(self):
        """Test with enough values to require three batches."""
        Base = declarative_base()

        class TestModel(Base):
            __tablename__ = "test"
            id = Column(Integer, primary_key=True)

        values = list(range(_SQLITE_MAX_BIND_VARS * 2 + 100))
        condition = _batched_in_condition(TestModel.id, values)

        condition_str = str(condition)
        assert "OR" in condition_str
        # Should have at least 3 IN clauses
        assert condition_str.count("IN") >= 3

    def test_batched_in_condition_empty_list(self):
        """Test with an empty list."""
        Base = declarative_base()

        class TestModel(Base):
            __tablename__ = "test"
            id = Column(Integer, primary_key=True)

        values = []
        condition = _batched_in_condition(TestModel.id, values)

        # Empty list should still generate valid SQL
        condition_str = str(condition)
        assert "IN" in condition_str

    def test_batched_in_condition_multiple_columns(self):
        """Test combining multiple batched conditions with AND logic."""
        from sqlalchemy import String, and_

        Base = declarative_base()

        class TestModel(Base):
            __tablename__ = "test"
            id = Column(Integer, primary_key=True)
            name = Column(String)
            email = Column(String)

        # Create multiple large value lists for different columns
        num_values = (_SQLITE_MAX_BIND_VARS * 2) + 100
        id_values = list(range(num_values))
        name_values = [f"name_{i}" for i in range(num_values)]
        email_values = [f"email_{i}@test.com" for i in range(num_values)]

        # Create batched conditions for each column
        id_condition = _batched_in_condition(TestModel.id, id_values)
        name_condition = _batched_in_condition(TestModel.name, name_values)
        email_condition = _batched_in_condition(TestModel.email, email_values)

        # Combine with AND (simulating real query behavior)
        combined_condition = and_(id_condition, name_condition, email_condition)
        combined_str = str(combined_condition)

        # Verify all three columns are present in the query
        assert "id" in combined_str.lower()
        assert "name" in combined_str.lower()
        assert "email" in combined_str.lower()

        # Verify OR clauses are present (batching is active)
        assert combined_str.count("OR") >= 3  # At least one OR per batched column

        # Verify AND combines the conditions
        assert "AND" in combined_str

        # Verify `id` count matches expected batches
        expected_id_batches = (num_values + _SQLITE_MAX_BIND_VARS - 1) // _SQLITE_MAX_BIND_VARS
        actual_id_batches = combined_str.count("id IN")
        assert actual_id_batches == expected_id_batches

        # Verify `name` count matches expected batches
        expected_name_batches = (num_values + _SQLITE_MAX_BIND_VARS - 1) // _SQLITE_MAX_BIND_VARS
        actual_name_batches = combined_str.count("name IN")
        assert actual_name_batches == expected_name_batches

        # Verify `email` count matches expected batches
        expected_email_batches = (num_values + _SQLITE_MAX_BIND_VARS - 1) // _SQLITE_MAX_BIND_VARS
        actual_email_batches = combined_str.count("email IN")
        assert actual_email_batches == expected_email_batches


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

    def test_get_message_pieces_combines_filters_correctly(self, sqlite_instance: MemoryInterface):
        """Test that multiple filters can be combined (e.g., prompt_ids AND role)."""
        # Create message pieces with different roles
        num_pieces = 50
        user_pieces = [_create_message_piece() for _ in range(num_pieces)]
        for piece in user_pieces:
            piece.role = "user"

        assistant_pieces = [_create_message_piece() for _ in range(num_pieces)]
        for piece in assistant_pieces:
            piece.role = "assistant"

        all_pieces = user_pieces + assistant_pieces
        sqlite_instance.add_message_pieces_to_memory(message_pieces=all_pieces)

        # Query with both prompt_ids AND role filter
        user_ids = [piece.id for piece in user_pieces]
        results = sqlite_instance.get_message_pieces(prompt_ids=user_ids, role="user")

        # Should return only user pieces (intersection of both filters)
        assert len(results) == num_pieces
        assert all(r.role == "user" for r in results)

        # Query with role filter and a subset of IDs
        subset_ids = user_ids[:10]
        results = sqlite_instance.get_message_pieces(prompt_ids=subset_ids, role="user")
        assert len(results) == 10

    def test_get_message_pieces_multiple_large_params_simultaneously(self, sqlite_instance: MemoryInterface):
        """Test batching with multiple parameters exceeding batch limit simultaneously."""
        # Create enough pieces to exceed batch limit with unique values
        num_pieces = _SQLITE_MAX_BIND_VARS + 200
        pieces = [
            _create_message_piece(original_value=f"original_value_{i}") for i in range(num_pieces)
        ]

        # Add to memory
        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        # Get all stored pieces to extract their IDs and SHA256 hashes
        stored_pieces = sqlite_instance.get_message_pieces()
        assert len(stored_pieces) >= num_pieces

        # Extract multiple large parameter lists
        all_ids = [piece.id for piece in stored_pieces[:num_pieces]]
        all_original_values = [piece.original_value for piece in stored_pieces[:num_pieces]]
        all_sha256 = [piece.converted_value_sha256 for piece in stored_pieces[:num_pieces]]

        # Query with multiple large parameters simultaneously
        # This tests that ALL parameters are batched correctly, not just one
        results = sqlite_instance.get_message_pieces(
            prompt_ids=all_ids,
            original_values=all_original_values,
            converted_value_sha256=all_sha256,
        )

        # Should return all pieces that match ALL conditions (intersection)
        assert len(results) == num_pieces, (
            f"Expected {num_pieces} results when filtering with multiple large parameters, "
            f"got {len(results)}"
        )

        # Verify all returned pieces match all filter criteria
        result_ids = {r.id for r in results}
        result_original_values = {r.original_value for r in results}
        result_sha256 = {r.converted_value_sha256 for r in results}

        assert result_ids == set(all_ids), "Returned IDs don't match filter"
        assert result_original_values == set(all_original_values), "Returned original_values don't match filter"
        assert result_sha256 == set(all_sha256), "Returned SHA256 hashes don't match filter"

    def test_get_message_pieces_multiple_batched_params_with_query_spy(self, sqlite_instance: MemoryInterface):
        """Test that batching generates correct queries when multiple params exceed limit."""
        # Create pieces exceeding batch limit
        num_pieces = _SQLITE_MAX_BIND_VARS + 100
        pieces = [
            _create_message_piece(original_value=f"value_{i}") for i in range(num_pieces)
        ]
        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        # Get stored pieces
        stored_pieces = sqlite_instance.get_message_pieces()
        all_ids = [piece.id for piece in stored_pieces[:num_pieces]]
        all_original_values = [piece.original_value for piece in stored_pieces[:num_pieces]]

        # Mock _query_entries to track how it's called
        original_query = sqlite_instance._query_entries
        call_count = 0
        captured_conditions = []

        def spy_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "conditions" in kwargs and kwargs["conditions"] is not None:
                captured_conditions.append(str(kwargs["conditions"]))
            return original_query(*args, **kwargs)

        with patch.object(sqlite_instance, "_query_entries", side_effect=spy_query):
            results = sqlite_instance.get_message_pieces(
                prompt_ids=all_ids, original_values=all_original_values
            )

        # Should get all results despite batching
        assert len(results) == num_pieces

        # Should have been called (could be 1 call with OR conditions)
        assert call_count >= 1

        # Verify query conditions include both filters
        if captured_conditions:
            combined_conditions = " ".join(captured_conditions)
            # Both column filters should be present in the query
            assert "id" in combined_conditions.lower() or "prompt" in combined_conditions.lower()
            assert "original_value" in combined_conditions.lower()

    def test_get_message_pieces_triple_large_params_preserves_intersection(self, sqlite_instance: MemoryInterface):
        """Test that filtering with 3 large parameter lists returns correct intersection."""
        # Create a large set of pieces
        total_pieces = _SQLITE_MAX_BIND_VARS + 150
        pieces = [
            _create_message_piece(
                conversation_id=str(uuid.uuid4()), original_value=f"content_{i}"
            )
            for i in range(total_pieces)
        ]
        sqlite_instance.add_message_pieces_to_memory(message_pieces=pieces)

        # Get stored pieces
        stored_pieces = sqlite_instance.get_message_pieces()

        # Create three overlapping large filter lists
        # List 1: All IDs
        filter_ids = [p.id for p in stored_pieces[:total_pieces]]

        # List 2: All original values
        filter_original_values = [p.original_value for p in stored_pieces[:total_pieces]]

        # List 3: Subset of SHA256 hashes (to test intersection)
        subset_size = _SQLITE_MAX_BIND_VARS + 50
        filter_sha256 = [p.converted_value_sha256 for p in stored_pieces[:subset_size]]

        # Query with all three large parameters
        results = sqlite_instance.get_message_pieces(
            prompt_ids=filter_ids,
            original_values=filter_original_values,
            converted_value_sha256=filter_sha256,
        )

        # Should return only the intersection (subset_size items)
        assert (
            len(results) == subset_size
        ), f"Expected {subset_size} results from intersection, got {len(results)}"

        # Verify all results have SHA256 in the filter list
        result_sha256 = {r.converted_value_sha256 for r in results}
        assert result_sha256.issubset(set(filter_sha256)), "Results contain unexpected SHA256 values"
