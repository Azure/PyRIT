# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import MagicMock

import pytest

from pyrit.executor.attack import ObjectiveEvaluator
from pyrit.models.score import Score
from pyrit.score import Scorer


@pytest.fixture
def mock_scorer() -> MagicMock:
    """Create a mock scorer for testing."""
    scorer = MagicMock(spec=Scorer)
    scorer.get_identifier.return_value = {"__type__": "MockScorer", "__module__": "test_module"}
    return scorer


@pytest.fixture
def true_false_score_true() -> Score:
    """Create a true/false score with value 'true'."""
    return Score(
        score_type="true_false",
        score_value="true",
        score_category=["test"],
        score_value_description="Test true score",
        score_rationale="This is a true score rationale",
        score_metadata={},
        prompt_request_response_id=str(uuid.uuid4()),
        scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
    )


@pytest.fixture
def true_false_score_false() -> Score:
    """Create a true/false score with value 'false'."""
    return Score(
        score_type="true_false",
        score_value="false",
        score_category=["test"],
        score_value_description="Test false score",
        score_rationale="This is a false score rationale",
        score_metadata={},
        prompt_request_response_id=str(uuid.uuid4()),
        scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
    )


@pytest.fixture
def float_scale_score_high() -> Score:
    """Create a float scale score with high value."""
    return Score(
        score_type="float_scale",
        score_value="0.9",
        score_category=["test"],
        score_value_description="High float score",
        score_rationale="This is a high score rationale",
        score_metadata={},
        prompt_request_response_id=str(uuid.uuid4()),
        scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
    )


@pytest.fixture
def float_scale_score_low() -> Score:
    """Create a float scale score with low value."""
    return Score(
        score_type="float_scale",
        score_value="0.3",
        score_category=["test"],
        score_value_description="Low float score",
        score_rationale="This is a low score rationale",
        score_metadata={},
        prompt_request_response_id=str(uuid.uuid4()),
        scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
    )


class TestObjectiveEvaluatorInitialization:
    """Tests for ObjectiveEvaluator initialization and configuration."""

    def test_init_with_default_parameters(self, mock_scorer: MagicMock):
        """Test that ObjectiveEvaluator initializes correctly with default parameters."""
        evaluator = ObjectiveEvaluator(scorer=mock_scorer)

        assert evaluator._scorer == mock_scorer
        assert evaluator._use_score_as_feedback is True  # Default value
        assert evaluator._successful_objective_threshold == 0.8  # Default value

    def test_init_with_custom_parameters(self, mock_scorer: MagicMock):
        """Test that ObjectiveEvaluator initializes correctly with custom parameters."""
        evaluator = ObjectiveEvaluator(
            scorer=mock_scorer, use_score_as_feedback=False, successful_objective_threshold=0.6
        )

        assert evaluator._scorer == mock_scorer
        assert evaluator._use_score_as_feedback is False
        assert evaluator._successful_objective_threshold == 0.6

    @pytest.mark.parametrize(
        "threshold,expected_error",
        [
            (-0.1, "successful_objective_threshold must be between 0.0 and 1.0 \\(inclusive\\)"),
            (1.1, "successful_objective_threshold must be between 0.0 and 1.0 \\(inclusive\\)"),
            (-1.0, "successful_objective_threshold must be between 0.0 and 1.0 \\(inclusive\\)"),
            (2.0, "successful_objective_threshold must be between 0.0 and 1.0 \\(inclusive\\)"),
        ],
    )
    def test_init_with_invalid_threshold_raises_error(
        self, mock_scorer: MagicMock, threshold: float, expected_error: str
    ):
        """Test that invalid threshold values raise ValueError during initialization."""
        with pytest.raises(ValueError, match=expected_error):
            ObjectiveEvaluator(scorer=mock_scorer, successful_objective_threshold=threshold)

    @pytest.mark.parametrize(
        "threshold",
        [0.0, 0.5, 1.0],  # Valid edge cases
    )
    def test_init_with_edge_case_thresholds(self, mock_scorer: MagicMock, threshold: float):
        """Test that edge case threshold values are accepted."""
        evaluator = ObjectiveEvaluator(scorer=mock_scorer, successful_objective_threshold=threshold)
        assert evaluator._successful_objective_threshold == threshold


class TestIsObjectiveAchieved:
    """Tests for the is_objective_achieved method."""

    def test_is_objective_achieved_with_none_score(self, mock_scorer: MagicMock):
        """Test that None score returns False for objective achievement."""
        evaluator = ObjectiveEvaluator(scorer=mock_scorer)

        assert evaluator.is_objective_achieved(score=None) is False

    @pytest.mark.parametrize(
        "score_value,expected_result",
        [
            ("true", True),
            ("false", False),
            ("True", True),  # Test case sensitivity
            ("False", False),
            ("TRUE", True),
            ("FALSE", False),
        ],
    )
    def test_is_objective_achieved_with_true_false_scores(
        self, mock_scorer: MagicMock, score_value: str, expected_result: bool
    ):
        """Test objective achievement with various true/false score values."""
        evaluator = ObjectiveEvaluator(scorer=mock_scorer)

        score = Score(
            score_type="true_false",
            score_value=score_value,
            score_category=["test"],
            score_value_description="Test score",
            score_rationale="Test rationale",
            score_metadata={},
            prompt_request_response_id=str(uuid.uuid4()),
            scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
        )

        assert evaluator.is_objective_achieved(score=score) == expected_result

    @pytest.mark.parametrize(
        "score_value,threshold,expected_result",
        [
            ("0.9", 0.8, True),  # Above threshold
            ("0.8", 0.8, True),  # Equal to threshold
            ("0.7", 0.8, False),  # Below threshold
            ("1.0", 0.8, True),  # Maximum value
            ("0.0", 0.8, False),  # Minimum value
            ("0.5", 0.5, True),  # Equal to custom threshold
            ("0.49", 0.5, False),  # Just below custom threshold
        ],
    )
    def test_is_objective_achieved_with_float_scale_scores(
        self, mock_scorer: MagicMock, score_value: str, threshold: float, expected_result: bool
    ):
        """Test objective achievement with float scale scores and different thresholds."""
        evaluator = ObjectiveEvaluator(scorer=mock_scorer, successful_objective_threshold=threshold)

        score = Score(
            score_type="float_scale",
            score_value=score_value,
            score_category=["test"],
            score_value_description="Test score",
            score_rationale="Test rationale",
            score_metadata={},
            prompt_request_response_id=str(uuid.uuid4()),
            scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
        )

        assert evaluator.is_objective_achieved(score=score) == expected_result


class TestGetFeedback:
    """Tests for the get_feedback method."""

    def test_get_feedback_returns_rationale_when_enabled(self, mock_scorer: MagicMock, true_false_score_true: Score):
        """Test that feedback returns score rationale when enabled."""
        evaluator = ObjectiveEvaluator(scorer=mock_scorer, use_score_as_feedback=True)

        feedback = evaluator.get_feedback(score=true_false_score_true)

        assert feedback == true_false_score_true.score_rationale
        assert feedback == "This is a true score rationale"

    def test_get_feedback_returns_none_when_disabled(self, mock_scorer: MagicMock, true_false_score_true: Score):
        """Test that feedback returns None when use_score_as_feedback is False."""
        evaluator = ObjectiveEvaluator(scorer=mock_scorer, use_score_as_feedback=False)

        feedback = evaluator.get_feedback(score=true_false_score_true)

        assert feedback is None

    @pytest.mark.parametrize(
        "score_fixture,expected_rationale",
        [
            ("true_false_score_true", "This is a true score rationale"),
            ("true_false_score_false", "This is a false score rationale"),
            ("float_scale_score_high", "This is a high score rationale"),
            ("float_scale_score_low", "This is a low score rationale"),
        ],
    )
    def test_get_feedback_with_different_score_types(
        self, mock_scorer: MagicMock, score_fixture: str, expected_rationale: str, request
    ):
        """Test that feedback works correctly with different score types."""
        score = request.getfixturevalue(score_fixture)
        evaluator = ObjectiveEvaluator(scorer=mock_scorer, use_score_as_feedback=True)

        feedback = evaluator.get_feedback(score=score)

        assert feedback == expected_rationale

    def test_get_feedback_with_empty_rationale(self, mock_scorer: MagicMock):
        """Test feedback behavior when score has empty rationale."""
        score = Score(
            score_type="true_false",
            score_value="true",
            score_category=["test"],
            score_value_description="Test score",
            score_rationale="",  # Empty rationale
            score_metadata={},
            prompt_request_response_id=str(uuid.uuid4()),
            scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
        )

        evaluator = ObjectiveEvaluator(scorer=mock_scorer, use_score_as_feedback=True)
        feedback = evaluator.get_feedback(score=score)

        assert feedback == ""  # Should return empty string, not None


class TestEdgeCases:
    """Integration tests combining multiple ObjectiveEvaluator features."""

    def test_evaluate_true_false_score_with_feedback(self, mock_scorer: MagicMock, true_false_score_true: Score):
        """Test complete evaluation flow for true/false score with feedback enabled."""
        evaluator = ObjectiveEvaluator(scorer=mock_scorer, use_score_as_feedback=True)

        # Check if objective is achieved
        is_achieved = evaluator.is_objective_achieved(score=true_false_score_true)
        assert is_achieved is True

        # Get feedback
        feedback = evaluator.get_feedback(score=true_false_score_true)
        assert feedback == "This is a true score rationale"

        # Check scorer type
        assert evaluator.scorer_type == "MockScorer"

    def test_evaluate_float_scale_score_near_threshold(self, mock_scorer: MagicMock):
        """Test evaluation of float scores near the threshold boundary."""
        evaluator = ObjectiveEvaluator(
            scorer=mock_scorer, use_score_as_feedback=True, successful_objective_threshold=0.75
        )

        # Test score just above threshold
        score_above = Score(
            score_type="float_scale",
            score_value="0.751",
            score_category=["test"],
            score_value_description="Just above threshold",
            score_rationale="Score is slightly above the threshold",
            score_metadata={},
            prompt_request_response_id=str(uuid.uuid4()),
            scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
        )

        assert evaluator.is_objective_achieved(score=score_above) is True
        assert evaluator.get_feedback(score=score_above) == "Score is slightly above the threshold"

        # Test score just below threshold
        score_below = Score(
            score_type="float_scale",
            score_value="0.749",
            score_category=["test"],
            score_value_description="Just below threshold",
            score_rationale="Score is slightly below the threshold",
            score_metadata={},
            prompt_request_response_id=str(uuid.uuid4()),
            scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
        )

        assert evaluator.is_objective_achieved(score=score_below) is False
        assert evaluator.get_feedback(score=score_below) == "Score is slightly below the threshold"

    @pytest.mark.parametrize(
        "score_values,threshold,expected_achievements",
        [
            (["0.3", "0.5", "0.7", "0.9"], 0.6, [False, False, True, True]),
            (["0.1", "0.4", "0.8", "1.0"], 0.8, [False, False, True, True]),
            (["0.0", "0.25", "0.5", "0.75"], 0.5, [False, False, True, True]),
        ],
    )
    def test_evaluate_score_progression(
        self, mock_scorer: MagicMock, score_values: list, threshold: float, expected_achievements: list
    ):
        """Test evaluation of a progression of scores simulating an attack improving over time."""
        evaluator = ObjectiveEvaluator(scorer=mock_scorer, successful_objective_threshold=threshold)

        for score_value, expected_achieved in zip(score_values, expected_achievements):
            score = Score(
                score_type="float_scale",
                score_value=score_value,
                score_category=["test"],
                score_value_description=f"Score: {score_value}",
                score_rationale=f"Rationale for score {score_value}",
                score_metadata={},
                prompt_request_response_id=str(uuid.uuid4()),
                scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
            )

            assert evaluator.is_objective_achieved(score=score) == expected_achieved
