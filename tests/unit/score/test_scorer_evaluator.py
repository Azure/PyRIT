# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pyrit.common.path import SCORER_EVALS_HARM_PATH, SCORER_EVALS_TRUE_FALSE_PATH
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
def sample_harm_csv_path():
    return f"{str(SCORER_EVALS_HARM_PATH)}/mini_hate_speech.csv"


@pytest.fixture
def sample_objective_csv_path():
    return f"{str(SCORER_EVALS_TRUE_FALSE_PATH)}/mini_refusal.csv"


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


def test_get_scorer_metrics_harm(tmp_path, mock_harm_scorer):
    # Create a fake metrics file
    metrics = HarmScorerMetrics(
        mean_absolute_error=0.1,
        mae_standard_error=0.01,
        t_statistic=1.0,
        p_value=0.05,
        krippendorff_alpha_combined=0.8,
        krippendorff_alpha_humans=0.7,
        krippendorff_alpha_model=0.9,
    )
    metrics_path = tmp_path / "metrics.json"
    with open(metrics_path, "w") as f:
        f.write(metrics.to_json())
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    # Patch _get_metrics_path to return our temp file
    with patch.object(evaluator, "_get_metrics_path", return_value=metrics_path):
        loaded = evaluator.get_scorer_metrics("any_dataset")
        assert loaded == metrics

    with patch.object(evaluator, "_get_metrics_path", return_value=tmp_path / "does_not_exist.json"):
        with pytest.raises(FileNotFoundError):
            evaluator.get_scorer_metrics("any_dataset")


def test_get_scorer_metrics_harm_with_results_folder_search(tmp_path, mock_harm_scorer):
    """Test that get_scorer_metrics finds metrics in results folders when default location fails."""
    evaluator = HarmScorerEvaluator(mock_harm_scorer)

    # Create metrics in a results folder
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    metrics = HarmScorerMetrics(
        mean_absolute_error=0.1,
        mae_standard_error=0.01,
        t_statistic=1.0,
        p_value=0.05,
        krippendorff_alpha_combined=0.8,
    )
    metrics_filename = "test_dataset_MagicMock_metrics.json"
    metrics_path = results_dir / metrics_filename
    with open(metrics_path, "w") as f:
        f.write(metrics.to_json())

    # Mock _get_metrics_path to return non-existent default path
    default_path = tmp_path / "default" / "does_not_exist.json"

    with patch.object(evaluator, "_get_metrics_path", return_value=default_path):
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            loaded = evaluator.get_scorer_metrics("test_dataset")
            assert loaded == metrics


def test_get_scorer_metrics_objective_with_results_folder_search(tmp_path, mock_objective_scorer):
    """Test that get_scorer_metrics finds metrics in results folders when default location fails."""
    evaluator = ObjectiveScorerEvaluator(mock_objective_scorer)

    # Create metrics in a results folder
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    metrics = ObjectiveScorerMetrics(
        accuracy=0.9,
        accuracy_standard_error=0.05,
        f1_score=0.8,
        precision=0.85,
        recall=0.75,
    )
    metrics_filename = "test_dataset_MagicMock_metrics.json"
    metrics_path = results_dir / metrics_filename
    with open(metrics_path, "w") as f:
        f.write(metrics.to_json())

    # Mock _get_metrics_path to return non-existent default path
    default_path = tmp_path / "default" / "does_not_exist.json"

    with patch.object(evaluator, "_get_metrics_path", return_value=default_path):
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            loaded = evaluator.get_scorer_metrics("test_dataset")
            assert loaded == metrics


def test_get_scorer_metrics_no_metrics_found_error_message(tmp_path, mock_harm_scorer):
    """Test that get_scorer_metrics provides helpful error message when no metrics found."""
    evaluator = HarmScorerEvaluator(mock_harm_scorer)

    # Mock _get_metrics_path to return non-existent default path
    default_path = tmp_path / "default" / "does_not_exist.json"

    with patch.object(evaluator, "_get_metrics_path", return_value=default_path):
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with pytest.raises(FileNotFoundError) as exc_info:
                evaluator.get_scorer_metrics("test_dataset")

            error_msg = str(exc_info.value)
            assert "Metrics for dataset 'test_dataset' not found" in error_msg
            assert "Default location:" in error_msg
            assert "Results folders:" in error_msg


def test_get_scorer_metrics_objective(tmp_path, mock_objective_scorer):
    metrics = ObjectiveScorerMetrics(
        accuracy=0.9,
        accuracy_standard_error=0.05,
        f1_score=0.8,
        precision=0.85,
        recall=0.75,
    )
    metrics_path = tmp_path / "metrics.json"
    with open(metrics_path, "w") as f:
        f.write(metrics.to_json())
    evaluator = ObjectiveScorerEvaluator(mock_objective_scorer)

    with patch.object(evaluator, "_get_metrics_path", return_value=metrics_path):
        loaded = evaluator.get_scorer_metrics("any_dataset")
        assert loaded == metrics

    with patch.object(evaluator, "_get_metrics_path", return_value=tmp_path / "does_not_exist.json"):
        with pytest.raises(FileNotFoundError):
            evaluator.get_scorer_metrics("any_dataset")


@pytest.mark.asyncio
@patch(
    "pyrit.score.scorer_evaluation.scorer_evaluator.HarmScorerEvaluator.run_evaluation_async", new_callable=AsyncMock
)
async def test_run_evaluation_from_csv_async_harm(mock_run_eval, sample_harm_csv_path, mock_harm_scorer):
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    expected_metrics = HarmScorerMetrics(
        mean_absolute_error=0.1,
        mae_standard_error=0.01,
        t_statistic=1.0,
        p_value=0.05,
        krippendorff_alpha_combined=0.8,
        krippendorff_alpha_humans=0.7,
        krippendorff_alpha_model=0.9,
    )
    mock_run_eval.return_value = expected_metrics

    result = await evaluator.run_evaluation_from_csv_async(
        csv_path=sample_harm_csv_path,
        assistant_response_col_name="assistant_response",
        human_label_col_names=["human_score_1", "human_score_2", "human_score_3"],
        objective_or_harm_col_name="category",
        num_scorer_trials=2,
        save_results=False,
        dataset_name="SAMPLE_hate_speech",
    )

    assert result == expected_metrics
    mock_run_eval.assert_awaited_once()


@pytest.mark.asyncio
@patch(
    "pyrit.score.scorer_evaluation.scorer_evaluator.ObjectiveScorerEvaluator.run_evaluation_async",
    new_callable=AsyncMock,
)
async def test_run_evaluation_from_csv_async_objective(mock_run_eval, sample_objective_csv_path, mock_objective_scorer):
    evaluator = ObjectiveScorerEvaluator(mock_objective_scorer)
    expected_metrics = ObjectiveScorerMetrics(
        accuracy=0.9,
        accuracy_standard_error=0.05,
        f1_score=0.8,
        precision=0.85,
        recall=0.75,
    )
    mock_run_eval.return_value = expected_metrics

    result = await evaluator.run_evaluation_from_csv_async(
        csv_path=sample_objective_csv_path,
        assistant_response_col_name="assistant_response",
        human_label_col_names=["normalized_score"],
        objective_or_harm_col_name="objective",
        assistant_response_data_type_col_name="data_type",
        num_scorer_trials=2,
        save_results=False,
        dataset_name="SAMPLE_mixed_objective_refusal",
    )

    assert result == expected_metrics
    mock_run_eval.assert_awaited_once()


def test_save_model_scores_to_csv(tmp_path, mock_harm_scorer):
    """Test _save_model_scores_to_csv basic functionality without true_scores."""
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    objectives_or_harms = ["hate_speech", "hate_speech"]
    responses = ["resp1", "resp2"]
    all_model_scores = np.array([[1, 0], [0, 1]])
    file_path = tmp_path / "results.csv"

    evaluator._save_model_scores_to_csv(objectives_or_harms, responses, all_model_scores, file_path)

    df = pd.read_csv(file_path)
    assert list(df["objective_or_harm"]) == objectives_or_harms
    assert list(df["assistant_response"]) == responses
    assert "trial 1" in df.columns
    assert "trial 2" in df.columns
    assert "true_label" not in df.columns

    # Clean up the temporary file
    file_path.unlink()


def test_save_model_scores_to_csv_with_true_scores_array(tmp_path, mock_harm_scorer):
    """Test _save_model_scores_to_csv with true_scores as numpy array."""
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    objectives_or_harms = ["hate_speech", "hate_speech"]
    responses = ["resp1", "resp2"]
    all_model_scores = np.array([[1, 0], [0, 1]])
    true_scores = np.array([0.8, 0.2])
    file_path = tmp_path / "results.csv"

    evaluator._save_model_scores_to_csv(objectives_or_harms, responses, all_model_scores, file_path, true_scores)

    df = pd.read_csv(file_path)
    assert list(df["objective_or_harm"]) == objectives_or_harms
    assert list(df["assistant_response"]) == responses
    assert "trial 1" in df.columns
    assert "trial 2" in df.columns
    assert "true_label" in df.columns
    assert list(df["true_label"]) == [0.8, 0.2]

    # Clean up the temporary file
    file_path.unlink()


def test_save_model_scores_to_csv_with_true_scores_scalar(tmp_path, mock_harm_scorer):
    """Test _save_model_scores_to_csv with true_scores as scalar value."""
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    objectives_or_harms = ["hate_speech", "hate_speech"]
    responses = ["resp1", "resp2"]
    all_model_scores = np.array([[1, 0], [0, 1]])
    true_scores = 0.5  # scalar value
    file_path = tmp_path / "results.csv"

    evaluator._save_model_scores_to_csv(objectives_or_harms, responses, all_model_scores, file_path, true_scores)

    df = pd.read_csv(file_path)
    assert list(df["objective_or_harm"]) == objectives_or_harms
    assert list(df["assistant_response"]) == responses
    assert "trial 1" in df.columns
    assert "trial 2" in df.columns
    assert "true_label" in df.columns
    assert list(df["true_label"]) == [0.5, 0.5]  # scalar repeated for each response

    # Clean up the temporary file
    file_path.unlink()


def test_save_model_scores_to_csv_with_true_scores_list(tmp_path, mock_harm_scorer):
    """Test _save_model_scores_to_csv with true_scores as list."""
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    objectives_or_harms = ["hate_speech", "hate_speech"]
    responses = ["resp1", "resp2"]
    all_model_scores = np.array([[1, 0], [0, 1]])
    true_scores = [0.9, 0.1]  # list
    file_path = tmp_path / "results.csv"

    evaluator._save_model_scores_to_csv(objectives_or_harms, responses, all_model_scores, file_path, true_scores)

    df = pd.read_csv(file_path)
    assert "true_label" in df.columns
    assert list(df["true_label"]) == [0.9, 0.1]

    # Clean up the temporary file
    file_path.unlink()


def test_save_model_scores_to_csv_column_order(tmp_path, mock_harm_scorer):
    """Test that true_label column is added after all trial columns."""
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    objectives_or_harms = ["hate_speech"]
    responses = ["resp1"]
    all_model_scores = np.array([[1], [0], [1]])  # 3 trials, 1 response
    true_scores = [0.5]
    file_path = tmp_path / "results.csv"

    evaluator._save_model_scores_to_csv(objectives_or_harms, responses, all_model_scores, file_path, true_scores)

    df = pd.read_csv(file_path)
    expected_columns = ["objective_or_harm", "assistant_response", "trial 1", "trial 2", "trial 3", "true_label"]
    assert list(df.columns) == expected_columns


def test_get_metrics_path_and_csv_path_harm(mock_harm_scorer):
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    dataset_name = "SAMPLE_harm"
    expected_metrics_path = Path(SCORER_EVALS_HARM_PATH) / f"{dataset_name}_MagicMock_metrics.json"
    expected_csv_path = Path(SCORER_EVALS_HARM_PATH) / f"{dataset_name}_MagicMock_scoring_results.csv"
    metrics_path = evaluator._get_metrics_path(dataset_name)
    csv_path = evaluator._get_csv_results_path(dataset_name)
    assert metrics_path == expected_metrics_path
    assert csv_path == expected_csv_path


def test_get_metrics_path_and_csv_path_objective(mock_objective_scorer):
    evaluator = ObjectiveScorerEvaluator(mock_objective_scorer)
    dataset_name = "SAMPLE_objective"
    expected_metrics_path = Path(SCORER_EVALS_TRUE_FALSE_PATH) / f"{dataset_name}_MagicMock_metrics.json"
    expected_csv_path = Path(SCORER_EVALS_TRUE_FALSE_PATH) / f"{dataset_name}_MagicMock_scoring_results.csv"
    metrics_path = evaluator._get_metrics_path(dataset_name)
    csv_path = evaluator._get_csv_results_path(dataset_name)
    assert metrics_path == expected_metrics_path
    assert csv_path == expected_csv_path


@pytest.mark.asyncio
async def test_run_evaluation_async_harm(mock_harm_scorer):
    responses = [
        Message(message_pieces=[MessagePiece(role="assistant", original_value="test", original_value_data_type="text")])
    ]
    entry1 = HarmHumanLabeledEntry(responses, [0.1, 0.3], "hate_speech")
    entry2 = HarmHumanLabeledEntry(responses, [0.2, 0.6], "hate_speech")
    mock_dataset = HumanLabeledDataset(name="test_dataset", metrics_type=MetricsType.HARM, entries=[entry1, entry2])
    # Patch scorer to return fixed scores
    entry_values = [MagicMock(get_value=lambda: 0.2), MagicMock(get_value=lambda: 0.4)]
    mock_harm_scorer.score_prompts_batch_async = AsyncMock(return_value=entry_values)
    evaluator = HarmScorerEvaluator(mock_harm_scorer)
    metrics = await evaluator.run_evaluation_async(
        labeled_dataset=mock_dataset, num_scorer_trials=2, save_results=False
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
    mock_dataset = HumanLabeledDataset(name="test_dataset", metrics_type=MetricsType.OBJECTIVE, entries=[entry])
    # Patch scorer to return fixed scores
    mock_objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(
        return_value=[MagicMock(get_value=lambda: False)]
    )
    evaluator = ObjectiveScorerEvaluator(mock_objective_scorer)
    metrics = await evaluator.run_evaluation_async(
        labeled_dataset=mock_dataset, num_scorer_trials=2, save_results=False
    )
    assert mock_objective_scorer._memory.add_message_to_memory.call_count == 1
    assert isinstance(metrics, ObjectiveScorerMetrics)
    assert metrics.accuracy == 0.0
    assert metrics.accuracy_standard_error == 0.0


def test_compute_objective_metrics_perfect_agreement(mock_objective_scorer):
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    # 2 responses, 3 human scores each, all agree (all 1s), model also all 1s
    all_human_scores = np.array([[1, 1], [1, 1], [1, 1]])
    all_model_scores = np.array([[1, 1], [1, 1]])
    metrics = evaluator._compute_objective_metrics(all_human_scores=all_human_scores, all_model_scores=all_model_scores)
    assert metrics.accuracy == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_compute_objective_metrics_partial_agreement(mock_objective_scorer):
    evaluator = ObjectiveScorerEvaluator(scorer=mock_objective_scorer)
    # 2 responses, 3 human scores each, mixed labels, model gets one right, one wrong
    all_human_scores = np.array([[1, 0], [1, 0], [0, 1]])  # gold: [1, 0]
    all_model_scores = np.array([[1, 1]])
    metrics = evaluator._compute_objective_metrics(all_human_scores=all_human_scores, all_model_scores=all_model_scores)
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
    metrics = evaluator._compute_harm_metrics(all_human_scores=all_human_scores, all_model_scores=all_model_scores)
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
    metrics = evaluator._compute_harm_metrics(all_human_scores=all_human_scores, all_model_scores=all_model_scores)
    assert np.isclose(metrics.mean_absolute_error, 0.1)
