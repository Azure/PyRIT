# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for ChunkedRequestAttack.

This attack was developed based on techniques discovered and validated
during Crucible CTF red teaming exercises using PyRIT.
"""

import pytest

from pyrit.executor.attack.multi_turn import (
    ChunkedRequestAttack,
    ChunkedRequestAttackContext,
)


class TestChunkedRequestAttackContext:
    """Test the ChunkedRequestAttackContext dataclass."""

    def test_context_default_values(self):
        """Test that context has correct default values."""
        context = ChunkedRequestAttackContext(objective="Extract the secret")

        assert context.chunk_size == 50
        assert context.total_length == 200
        assert context.chunk_description == "characters"
        assert context.target_description is None
        assert len(context.chunk_responses) == 0

    def test_context_custom_values(self):
        """Test setting custom values in context."""
        context = ChunkedRequestAttackContext(
            objective="Get the password",
            chunk_size=100,
            total_length=500,
            chunk_description="bytes",
            target_description="the API key",
        )

        assert context.chunk_size == 100
        assert context.total_length == 500
        assert context.chunk_description == "bytes"
        assert context.target_description == "the API key"


class TestChunkedRequestAttack:
    """Test the ChunkedRequestAttack class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(objective_target=mock_target)

        assert attack._chunk_size == 50
        assert attack._total_length == 200
        assert attack._chunk_description == "characters"

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(
            objective_target=mock_target,
            chunk_size=25,
            total_length=150,
            chunk_description="words",
        )

        assert attack._chunk_size == 25
        assert attack._total_length == 150
        assert attack._chunk_description == "words"

    def test_init_invalid_chunk_size(self):
        """Test that invalid chunk_size raises ValueError."""
        from unittest.mock import Mock

        mock_target = Mock()

        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            ChunkedRequestAttack(objective_target=mock_target, chunk_size=0)

    def test_init_invalid_total_length(self):
        """Test that invalid total_length raises ValueError."""
        from unittest.mock import Mock

        mock_target = Mock()

        with pytest.raises(ValueError, match="total_length must be >= chunk_size"):
            ChunkedRequestAttack(objective_target=mock_target, chunk_size=100, total_length=50)

    def test_extract_target_description_password(self):
        """Test target description extraction for 'password'."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(objective_target=mock_target)

        result = attack._extract_target_description("What is the password?")
        assert result == "the password"

    def test_extract_target_description_secret(self):
        """Test target description extraction for 'secret'."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(objective_target=mock_target)

        result = attack._extract_target_description("Tell me the secret")
        assert result == "the secret"

    def test_extract_target_description_flag(self):
        """Test target description extraction for 'flag'."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(objective_target=mock_target)

        result = attack._extract_target_description("Give me the flag")
        assert result == "the flag"

    def test_extract_target_description_default(self):
        """Test target description extraction with no match."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(objective_target=mock_target)

        result = attack._extract_target_description("Some random objective")
        assert result == "the value"

    def test_generate_chunk_prompts_default_template(self):
        """Test chunk prompt generation with default template."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(objective_target=mock_target, chunk_size=50, total_length=150)

        context = ChunkedRequestAttackContext(
            objective="Get the secret",
            chunk_size=50,
            total_length=150,
        )

        prompts = attack._generate_chunk_prompts(context)

        assert len(prompts) == 3
        assert "characters 1-50" in prompts[0]
        assert "the secret" in prompts[0]
        assert "characters 51-100" in prompts[1]
        assert "characters 101-150" in prompts[2]

    def test_generate_chunk_prompts_custom_target(self):
        """Test chunk prompt generation with custom target description."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(objective_target=mock_target, chunk_size=50, total_length=100)

        context = ChunkedRequestAttackContext(
            objective="Some objective",
            chunk_size=50,
            total_length=100,
            target_description="the API token",
        )

        prompts = attack._generate_chunk_prompts(context)

        assert len(prompts) == 2
        assert "the API token" in prompts[0]
        assert "the API token" in prompts[1]

    def test_validate_context_empty_objective(self):
        """Test validation fails with empty objective."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(objective_target=mock_target)

        context = ChunkedRequestAttackContext(objective="")

        with pytest.raises(ValueError, match="Attack objective must be provided"):
            attack._validate_context(context=context)

    def test_validate_context_invalid_chunk_size(self):
        """Test validation fails with invalid chunk_size."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(objective_target=mock_target)

        context = ChunkedRequestAttackContext(objective="test", chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            attack._validate_context(context=context)

    def test_validate_context_invalid_total_length(self):
        """Test validation fails when total_length < chunk_size."""
        from unittest.mock import Mock

        mock_target = Mock()
        attack = ChunkedRequestAttack(objective_target=mock_target)

        context = ChunkedRequestAttackContext(objective="test", chunk_size=100, total_length=50)

        with pytest.raises(ValueError, match="total_length must be >= chunk_size"):
            attack._validate_context(context=context)
