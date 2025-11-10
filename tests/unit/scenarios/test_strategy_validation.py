# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for strategy composition validation."""

import pytest

from pyrit.scenarios import EncodingStrategy, FoundryStrategy


class TestStrategyValidation:
    """Test validation of strategy compositions."""

    def test_encoding_validation_allows_single_strategy(self):
        """Test that encoding validation allows single strategies."""
        # Should not raise
        EncodingStrategy.validate_composition([EncodingStrategy.Base64])

    def test_encoding_validation_rejects_composition(self):
        """Test that encoding validation rejects composed strategies."""
        with pytest.raises(ValueError, match="EncodingStrategy does not support composition"):
            EncodingStrategy.validate_composition([EncodingStrategy.Base64, EncodingStrategy.ROT13])

    def test_foundry_validation_allows_single_strategy(self):
        """Test that foundry validation allows single strategies."""
        # Should not raise
        FoundryStrategy.validate_composition([FoundryStrategy.Base64])

    def test_foundry_validation_allows_converter_composition(self):
        """Test that foundry validation allows multiple converters."""
        # Should not raise
        FoundryStrategy.validate_composition([FoundryStrategy.Base64, FoundryStrategy.Atbash])

    def test_foundry_validation_allows_one_attack_with_converters(self):
        """Test that foundry validation allows one attack with converters."""
        # Should not raise
        FoundryStrategy.validate_composition(
            [FoundryStrategy.Base64, FoundryStrategy.Crescendo, FoundryStrategy.Atbash]
        )

    def test_foundry_validation_rejects_multiple_attacks(self):
        """Test that foundry validation rejects multiple attack strategies."""
        with pytest.raises(ValueError, match="Cannot compose multiple attack strategies"):
            FoundryStrategy.validate_composition([FoundryStrategy.Crescendo, FoundryStrategy.MultiTurn])

    def test_foundry_validation_rejects_attacks_with_converters_and_another_attack(self):
        """Test that foundry validation rejects multiple attacks even with converters."""
        with pytest.raises(ValueError, match="Cannot compose multiple attack strategies"):
            FoundryStrategy.validate_composition(
                [FoundryStrategy.Base64, FoundryStrategy.Crescendo, FoundryStrategy.MultiTurn]
            )
