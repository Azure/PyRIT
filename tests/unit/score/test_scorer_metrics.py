# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import pytest

from pyrit.score import HarmScorerMetrics, ObjectiveScorerMetrics


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
