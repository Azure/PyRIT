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
    TrueFalseScorer,
)


@pytest.fixture
def mock_harm_scorer():
    scorer = MagicMock(spec=FloatScaleScorer)
    scorer._memory = MagicMock()
    return scorer


@pytest.fixture
def mock_objective_scorer():
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer._memory = MagicMock()
    return scorer


def test_harm_metrics_to_json_and_from_json(tmp_path):
    metrics = HarmScorerMetrics(
        mean_absolute_error=0.1,
        mae_standard_error=0.01,
        t_statistic=1.0,
        p_value=0.05,
        krippendorff_alpha_combined=0.8,
        krippendorff_alpha_humans=0.7,
        krippendorff_alpha_model=0.9,
    )
    json_str = metrics.to_json()
    data = json.loads(json_str)
    assert data["mean_absolute_error"] == 0.1

    # Save to file and reload
    file_path = tmp_path / "metrics.json"
    with open(file_path, "w") as f:
        f.write(json_str)
    loaded = HarmScorerMetrics.from_json(str(file_path))
    assert loaded == metrics


def test_objective_metrics_to_json_and_from_json(tmp_path):
    metrics = ObjectiveScorerMetrics(
        accuracy=0.9,
        accuracy_standard_error=0.05,
        f1_score=0.8,
        precision=0.85,
        recall=0.75,
    )
    json_str = metrics.to_json()
    data = json.loads(json_str)
    assert data["accuracy"] == 0.9

    file_path = tmp_path / "metrics.json"
    with open(file_path, "w") as f:
        f.write(json_str)
    loaded = ObjectiveScorerMetrics.from_json(str(file_path))
    assert loaded == metrics


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
async def test_run_evaluation_async_harm(mock_harm_scorer):
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
    metrics = await evaluator.run_evaluation_async(
        labeled_dataset=mock_dataset, num_scorer_trials=2
    )
    assert mock_harm_scorer._memory.add_message_to_memory.call_count == 2
    assert isinstance(metrics, HarmScorerMetrics)
    assert metrics.mean_absolute_error == 0.0
    assert metrics.mae_standard_error == 0.0


@pytest.mark.asyncio
async def test_run_evaluation_async_objective(mock_objective_scorer):
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
    metrics = await evaluator.run_evaluation_async(
        labeled_dataset=mock_dataset, num_scorer_trials=2
    )
    assert mock_objective_scorer._memory.add_message_to_memory.call_count == 1
    assert isinstance(metrics, ObjectiveScorerMetrics)
    assert metrics.accuracy == 0.0
    assert metrics.accuracy_standard_error == 0.0


@pytest.mark.asyncio
async def test_run_evaluation_async_objective_returns_metrics(mock_objective_scorer):
    """Test that run_evaluation_async returns metrics without side effects."""
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

    metrics = await evaluator.run_evaluation_async(
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
    metrics = evaluator._compute_metrics(all_human_scores=all_human_scores, all_model_scores=all_model_scores)
    assert metrics.accuracy == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_compute_objective_metrics_partial_agreement(mock_objective_scorer):
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    # 2 responses, 3 human scores each, mixed labels, model gets one right, one wrong
    all_human_scores = np.array([[1, 0], [1, 0], [0, 1]])  # gold: [1, 0]
    all_model_scores = np.array([[1, 1]])
    metrics = evaluator._compute_metrics(all_human_scores=all_human_scores, all_model_scores=all_model_scores)
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
    metrics = evaluator._compute_metrics(all_human_scores=all_human_scores, all_model_scores=all_model_scores)
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
    metrics = evaluator._compute_metrics(all_human_scores=all_human_scores, all_model_scores=all_model_scores)
    assert np.isclose(metrics.mean_absolute_error, 0.1)
