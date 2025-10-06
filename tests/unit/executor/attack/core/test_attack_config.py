# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import MagicMock

from pyrit.executor.attack.core import AttackScoringConfig
from pyrit.score import Scorer
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer


class TestAttackScoringConfig:
    """Test AttackScoringConfig validation functionality."""

    def test_init_with_valid_objective_scorer(self):
        """Test initialization with a valid TrueFalseScorer for objective_scorer."""
        mock_scorer = MagicMock(spec=TrueFalseScorer)

        config = AttackScoringConfig(objective_scorer=mock_scorer)

        assert config.objective_scorer == mock_scorer

    def test_init_with_valid_refusal_scorer(self):
        """Test initialization with a valid TrueFalseScorer for refusal_scorer."""
        mock_scorer = MagicMock(spec=TrueFalseScorer)

        config = AttackScoringConfig(refusal_scorer=mock_scorer)

        assert config.refusal_scorer == mock_scorer

    def test_init_with_both_valid_scorers(self):
        """Test initialization with valid TrueFalseScorers for both objective and refusal scorers."""
        mock_objective_scorer = MagicMock(spec=TrueFalseScorer)
        mock_refusal_scorer = MagicMock(spec=TrueFalseScorer)

        config = AttackScoringConfig(objective_scorer=mock_objective_scorer, refusal_scorer=mock_refusal_scorer)

        assert config.objective_scorer == mock_objective_scorer
        assert config.refusal_scorer == mock_refusal_scorer

    def test_init_raises_error_for_non_true_false_objective_scorer(self):
        """Test that initialization raises ValueError for non-TrueFalseScorer objective_scorer."""
        mock_scorer = MagicMock(spec=Scorer)

        with pytest.raises(ValueError, match="Objective scorer must be a TrueFalseScorer"):
            AttackScoringConfig(objective_scorer=mock_scorer)

    def test_init_raises_error_for_non_true_false_refusal_scorer(self):
        """Test that initialization raises ValueError for non-TrueFalseScorer refusal_scorer."""
        mock_scorer = MagicMock(spec=Scorer)

        with pytest.raises(ValueError, match="Refusal scorer must be a TrueFalseScorer"):
            AttackScoringConfig(refusal_scorer=mock_scorer)

    def test_init_with_none_scorers(self):
        """Test initialization with None for both scorers (default behavior)."""
        config = AttackScoringConfig()

        assert config.objective_scorer is None
        assert config.refusal_scorer is None

    def test_init_with_invalid_threshold_too_high(self):
        """Test that initialization raises ValueError for threshold > 1.0."""
        with pytest.raises(
            ValueError, match="successful_objective_threshold must be between 0.0 and 1.0, got 1.5"
        ):
            AttackScoringConfig(successful_objective_threshold=1.5)

    def test_init_with_invalid_threshold_too_low(self):
        """Test that initialization raises ValueError for threshold < 0.0."""
        with pytest.raises(
            ValueError, match="successful_objective_threshold must be between 0.0 and 1.0, got -0.1"
        ):
            AttackScoringConfig(successful_objective_threshold=-0.1)

    def test_init_with_valid_threshold_boundaries(self):
        """Test initialization with valid threshold boundary values."""
        config_zero = AttackScoringConfig(successful_objective_threshold=0.0)
        config_one = AttackScoringConfig(successful_objective_threshold=1.0)

        assert config_zero.successful_objective_threshold == 0.0
        assert config_one.successful_objective_threshold == 1.0

    def test_init_with_auxiliary_scorers(self):
        """Test initialization with auxiliary scorers."""
        mock_aux_scorer_1 = MagicMock(spec=Scorer)
        mock_aux_scorer_2 = MagicMock(spec=Scorer)

        config = AttackScoringConfig(auxiliary_scorers=[mock_aux_scorer_1, mock_aux_scorer_2])

        assert len(config.auxiliary_scorers) == 2
        assert config.auxiliary_scorers[0] == mock_aux_scorer_1
        assert config.auxiliary_scorers[1] == mock_aux_scorer_2

    def test_init_with_use_score_as_feedback_false(self):
        """Test initialization with use_score_as_feedback set to False."""
        config = AttackScoringConfig(use_score_as_feedback=False)

        assert config.use_score_as_feedback is False
