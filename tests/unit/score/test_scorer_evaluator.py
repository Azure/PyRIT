# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from pyrit.models import Message, MessagePiece
from pyrit.score import (
    FloatScaleScorer,
    HarmHumanLabeledEntry,
    HarmScorerEvaluator,
    HarmScorerMetrics,
    HumanLabeledDataset,
    MetricsType,
    ObjectiveHumanLabeledEntry,
    ObjectiveScorerEvaluator,
    ObjectiveScorerMetrics,
    ScorerEvaluator,
    ScorerIdentifier,
    TrueFalseScorer,
)


@pytest.fixture
def mock_harm_scorer():
    scorer = MagicMock(spec=FloatScaleScorer)
    scorer._memory = MagicMock()
    scorer._memory.add_message_to_memory = MagicMock()
    scorer.scorer_identifier = ScorerIdentifier(
        type="FloatScaleScorer",
        system_prompt_template="test_system_prompt",
    )
    return scorer


@pytest.fixture
def mock_objective_scorer():
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer._memory = MagicMock()
    scorer._memory.add_message_to_memory = MagicMock()
    scorer.scorer_identifier = ScorerIdentifier(
        type="TrueFalseScorer",
        user_prompt_template="test_user_prompt",
    )
    return scorer


def test_from_scorer_harm(mock_harm_scorer):
    evaluator = ScorerEvaluator.from_scorer(mock_harm_scorer, metrics_type=MetricsType.HARM)
    assert isinstance(evaluator, HarmScorerEvaluator)
    evaluator2 = ScorerEvaluator.from_scorer(mock_harm_scorer)
    assert isinstance(evaluator2, HarmScorerEvaluator)


def test_from_scorer_objective(mock_objective_scorer):
    evaluator = ScorerEvaluator.from_scorer(mock_objective_scorer, metrics_type=MetricsType.OBJECTIVE)
    assert isinstance(evaluator, ObjectiveScorerEvaluator)
    evaluator2 = ScorerEvaluator.from_scorer(mock_objective_scorer)
    assert isinstance(evaluator2, ObjectiveScorerEvaluator)


@pytest.mark.asyncio
async def test__run_evaluation_async_harm(mock_harm_scorer):
    responses = [
        Message(message_pieces=[MessagePiece(role="assistant", original_value="test", original_value_data_type="text")])
    ]
    entry1 = HarmHumanLabeledEntry(responses, [0.1, 0.3], "hate_speech")
    entry2 = HarmHumanLabeledEntry(responses, [0.2, 0.6], "hate_speech")
    mock_dataset = HumanLabeledDataset(
        name="test_dataset", metrics_type=MetricsType.HARM, entries=[entry1, entry2], version="1.0"
    )
    # Patch scorer to return fixed scores
    entry_values = [MagicMock(get_value=lambda: 0.2), MagicMock(get_value=lambda: 0.4)]
    mock_harm_scorer.score_prompts_batch_async = AsyncMock(return_value=entry_values)
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    metrics = await evaluator._run_evaluation_async(
        labeled_dataset=mock_dataset, num_scorer_trials=2
    )
    assert mock_harm_scorer._memory.add_message_to_memory.call_count == 2
    assert isinstance(metrics, HarmScorerMetrics)
    assert metrics.mean_absolute_error == 0.0
    assert metrics.mae_standard_error == 0.0


@pytest.mark.asyncio
async def test__run_evaluation_async_objective(mock_objective_scorer):
    responses = [
        Message(message_pieces=[MessagePiece(role="assistant", original_value="test", original_value_data_type="text")])
    ]
    entry = ObjectiveHumanLabeledEntry(responses, [True], "Test objective")
    mock_dataset = HumanLabeledDataset(
        name="test_dataset", metrics_type=MetricsType.OBJECTIVE, entries=[entry], version="1.0"
    )
    # Patch scorer to return fixed scores
    mock_objective_scorer.score_prompts_batch_async = AsyncMock(
        return_value=[MagicMock(get_value=lambda: False)]
    )
    evaluator = ObjectiveScorerEvaluator(mock_objective_scorer)
    metrics = await evaluator._run_evaluation_async(
        labeled_dataset=mock_dataset, num_scorer_trials=2
    )
    assert mock_objective_scorer._memory.add_message_to_memory.call_count == 1
    assert isinstance(metrics, ObjectiveScorerMetrics)
    assert metrics.accuracy == 0.0
    assert metrics.accuracy_standard_error == 0.0


@pytest.mark.asyncio
async def test__run_evaluation_async_objective_returns_metrics(mock_objective_scorer):
    """Test that _run_evaluation_async returns metrics without side effects."""
    responses = [
        Message(message_pieces=[MessagePiece(role="assistant", original_value="test", original_value_data_type="text")])
    ]
    entry = ObjectiveHumanLabeledEntry(responses, [True], "Test objective")
    mock_dataset = HumanLabeledDataset(
        name="test_dataset", metrics_type=MetricsType.OBJECTIVE, entries=[entry], version="1.0"
    )
    mock_objective_scorer.score_prompts_batch_async = AsyncMock(
        return_value=[MagicMock(get_value=lambda: True)]
    )
    mock_objective_scorer.scorer_identifier = MagicMock()
    evaluator = ObjectiveScorerEvaluator(mock_objective_scorer)

    metrics = await evaluator._run_evaluation_async(
        labeled_dataset=mock_dataset, num_scorer_trials=1
    )
    
    # Verify metrics returned without registry writing
    assert metrics is not None
    assert isinstance(metrics, ObjectiveScorerMetrics)


def test_compute_objective_metrics_perfect_agreement(mock_objective_scorer):
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    # 2 responses, 3 human scores each, all agree (all 1s), model also all 1s
    all_human_scores = np.array([[1, 1], [1, 1], [1, 1]])
    all_model_scores = np.array([[1, 1], [1, 1]])
    metrics = evaluator._compute_metrics(
        all_human_scores=all_human_scores, all_model_scores=all_model_scores, num_scorer_trials=2
    )
    assert metrics.accuracy == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_compute_objective_metrics_partial_agreement(mock_objective_scorer):
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    # 2 responses, 3 human scores each, mixed labels, model gets one right, one wrong
    all_human_scores = np.array([[1, 0], [1, 0], [0, 1]])  # gold: [1, 0]
    all_model_scores = np.array([[1, 1]])
    metrics = evaluator._compute_metrics(
        all_human_scores=all_human_scores, all_model_scores=all_model_scores, num_scorer_trials=1
    )
    # gold: [1, 0], model: [1, 1]
    # TP=1 (first), FP=1 (second), TN=0, FN=0
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.f1_score == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))


def test_compute_harm_metrics_perfect_agreement(mock_harm_scorer):
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    # 2 responses, 3 human scores each, all agree, model matches exactly
    all_human_scores = np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
    # 2 model trials
    all_model_scores = np.array([[0.1, 0.2], [0.1, 0.2]])
    # Patch krippendorff.krippendorff_alpha to return 1.0 for all calls
    metrics = evaluator._compute_metrics(
        all_human_scores=all_human_scores, all_model_scores=all_model_scores, num_scorer_trials=2
    )
    assert metrics.mean_absolute_error == 0.0
    assert metrics.mae_standard_error == 0.0
    assert metrics.krippendorff_alpha_combined == 1.0
    assert metrics.krippendorff_alpha_humans == 1.0
    assert metrics.krippendorff_alpha_model == 1.0


def test_compute_harm_metrics_partial_agreement(mock_harm_scorer):
    evaluator = HarmScorerEvaluator(scorer=mock_harm_scorer)
    # 2 responses, 3 human scores each, model is off by 0.1 for each
    all_human_scores = np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
    all_model_scores = np.array([[0.2, 0.3], [0.2, 0.3]])
    metrics = evaluator._compute_metrics(
        all_human_scores=all_human_scores, all_model_scores=all_model_scores, num_scorer_trials=2
    )
    assert np.isclose(metrics.mean_absolute_error, 0.1)


@patch("pyrit.score.scorer_evaluation.scorer_evaluator.find_objective_metrics_by_hash")
def test_should_skip_evaluation_objective_found(mock_find, mock_objective_scorer, tmp_path):
    """Test skipping evaluation when existing objective metrics have sufficient trials."""
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    result_file = tmp_path / "test_results.jsonl"
    
    # Mock the compute_hash method on the scorer_identifier
    with patch.object(mock_objective_scorer.scorer_identifier, 'compute_hash', return_value="test_hash_123"):
        # Create expected metrics with same version and sufficient trials
        expected_metrics = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=3,
            accuracy=0.95,
            accuracy_standard_error=0.02,
            precision=0.96,
            recall=0.94,
            f1_score=0.95,
            num_scorer_trials=3,
            dataset_name="test_dataset",
            dataset_version="1.0",
        )
        mock_find.return_value = expected_metrics
        
        should_skip, result = evaluator._should_skip_evaluation(
            dataset_version="1.0",
            num_scorer_trials=3,
            harm_category=None,
            result_file_path=result_file,
        )
        
        assert should_skip is True
        assert result == expected_metrics
        mock_find.assert_called_once_with(
            file_path=result_file,
            hash="test_hash_123",
        )


@patch("pyrit.score.scorer_evaluation.scorer_evaluator.find_objective_metrics_by_hash")
def test_should_skip_evaluation_objective_not_found(mock_find, mock_objective_scorer, tmp_path):
    """Test when no existing objective metrics are found in registry."""
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    result_file = tmp_path / "test_results.jsonl"
    
    with patch.object(mock_objective_scorer.scorer_identifier, 'compute_hash', return_value="test_hash_123"):
        mock_find.return_value = None
        
        should_skip, result = evaluator._should_skip_evaluation(
            dataset_version="1.0",
            num_scorer_trials=3,
            harm_category=None,
            result_file_path=result_file,
        )
        
        assert should_skip is False
        assert result is None
        mock_find.assert_called_once_with(
            file_path=result_file,
            hash="test_hash_123",
        )


@patch("pyrit.score.scorer_evaluation.scorer_evaluator.find_objective_metrics_by_hash")
def test_should_skip_evaluation_version_changed_runs_evaluation(mock_find, mock_objective_scorer, tmp_path):
    """Test that different dataset_version triggers re-evaluation (replace existing)."""
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    result_file = tmp_path / "test_results.jsonl"
    
    with patch.object(mock_objective_scorer.scorer_identifier, 'compute_hash', return_value="test_hash_123"):
        # Metrics exist but with different dataset version
        existing_metrics = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=3,
            accuracy=0.95,
            accuracy_standard_error=0.02,
            precision=0.96,
            recall=0.94,
            f1_score=0.95,
            num_scorer_trials=3,
            dataset_name="test_dataset",
            dataset_version="2.0",  # Different version
        )
        mock_find.return_value = existing_metrics
        
        # When version differs, should NOT skip (run and replace)
        should_skip, result = evaluator._should_skip_evaluation(
            dataset_version="1.0",  # Looking for version 1.0
            num_scorer_trials=3,
            harm_category=None,
            result_file_path=result_file,
        )
        
        assert should_skip is False
        assert result is None


@patch("pyrit.score.scorer_evaluation.scorer_evaluator.find_objective_metrics_by_hash")
def test_should_skip_evaluation_fewer_trials_requested_skips(mock_find, mock_objective_scorer, tmp_path):
    """Test that requesting fewer trials than existing skips evaluation."""
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    result_file = tmp_path / "test_results.jsonl"
    
    with patch.object(mock_objective_scorer.scorer_identifier, 'compute_hash', return_value="test_hash_123"):
        # Metrics exist with more trials than requested
        existing_metrics = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=3,
            accuracy=0.95,
            accuracy_standard_error=0.02,
            precision=0.96,
            recall=0.94,
            f1_score=0.95,
            num_scorer_trials=5,  # Existing has 5 trials
            dataset_name="test_dataset",
            dataset_version="1.0",
        )
        mock_find.return_value = existing_metrics
        
        # Requesting only 3 trials - should skip since existing has more
        should_skip, result = evaluator._should_skip_evaluation(
            dataset_version="1.0",
            num_scorer_trials=3,  # Requesting fewer trials
            harm_category=None,
            result_file_path=result_file,
        )
        
        assert should_skip is True
        assert result == existing_metrics


@patch("pyrit.score.scorer_evaluation.scorer_evaluator.find_objective_metrics_by_hash")
def test_should_skip_evaluation_more_trials_requested_runs(mock_find, mock_objective_scorer, tmp_path):
    """Test that requesting more trials than existing triggers re-evaluation."""
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    result_file = tmp_path / "test_results.jsonl"
    
    with patch.object(mock_objective_scorer.scorer_identifier, 'compute_hash', return_value="test_hash_123"):
        # Metrics exist with fewer trials than requested
        existing_metrics = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=3,
            accuracy=0.95,
            accuracy_standard_error=0.02,
            precision=0.96,
            recall=0.94,
            f1_score=0.95,
            num_scorer_trials=2,  # Existing has only 2 trials
            dataset_name="test_dataset",
            dataset_version="1.0",
        )
        mock_find.return_value = existing_metrics
        
        # Requesting 5 trials - should NOT skip (run and replace)
        should_skip, result = evaluator._should_skip_evaluation(
            dataset_version="1.0",
            num_scorer_trials=5,  # Requesting more trials
            harm_category=None,
            result_file_path=result_file,
        )
        
        assert should_skip is False
        assert result is None


@patch("pyrit.score.scorer_evaluation.scorer_evaluator.find_harm_metrics_by_hash")
def test_should_skip_evaluation_harm_found(mock_find, mock_harm_scorer, tmp_path):
    """Test skipping evaluation when existing harm metrics have sufficient trials."""
    evaluator = HarmScorerEvaluator(scorer=mock_harm_scorer)
    result_file = tmp_path / "test_results.jsonl"
    
    with patch.object(mock_harm_scorer.scorer_identifier, 'compute_hash', return_value="test_hash_456"):
        # Create expected harm metrics
        expected_metrics = HarmScorerMetrics(
            num_responses=15,
            num_human_raters=4,
            mean_absolute_error=0.05,
            mae_standard_error=0.01,
            t_statistic=1.5,
            p_value=0.15,
            krippendorff_alpha_combined=0.85,
            krippendorff_alpha_humans=0.88,
            krippendorff_alpha_model=0.82,
            num_scorer_trials=3,
            dataset_name="harm_dataset",
            dataset_version="1.0",
            harm_category="hate_speech",
        )
        mock_find.return_value = expected_metrics
        
        should_skip, result = evaluator._should_skip_evaluation(
            dataset_version="1.0",
            num_scorer_trials=3,
            harm_category="hate_speech",
            result_file_path=result_file,
        )
        
        assert should_skip is True
        assert result == expected_metrics
        mock_find.assert_called_once_with(
            file_path=result_file,
            hash="test_hash_456",
        )


@patch("pyrit.score.scorer_evaluation.scorer_evaluator.find_harm_metrics_by_hash")
def test_should_skip_evaluation_harm_missing_category(mock_find, mock_harm_scorer, tmp_path):
    """Test that missing harm_category returns should not skip."""
    evaluator = HarmScorerEvaluator(scorer=mock_harm_scorer)
    result_file = tmp_path / "test_results.jsonl"
    
    with patch.object(mock_harm_scorer.scorer_identifier, 'compute_hash', return_value="test_hash_456"):
        # No harm_category provided
        should_skip, result = evaluator._should_skip_evaluation(
            dataset_version="1.0",
            num_scorer_trials=3,
            harm_category=None,
            result_file_path=result_file,
        )
        
        assert should_skip is False
        assert result is None
        mock_find.assert_not_called()


@patch("pyrit.score.scorer_evaluation.scorer_evaluator.find_objective_metrics_by_hash")
def test_should_skip_evaluation_exception_handling(mock_find, mock_objective_scorer, tmp_path):
    """Test that exceptions are caught and returns (False, None)."""
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    result_file = tmp_path / "test_results.jsonl"
    
    # Make compute_hash raise an exception
    with patch.object(
        mock_objective_scorer.scorer_identifier, 
        'compute_hash', 
        side_effect=Exception("Hash computation failed")
    ):
        should_skip, result = evaluator._should_skip_evaluation(
            dataset_version="1.0",
            num_scorer_trials=3,
            harm_category=None,
            result_file_path=result_file,
        )
        
        assert should_skip is False
        assert result is None
        mock_find.assert_not_called()

