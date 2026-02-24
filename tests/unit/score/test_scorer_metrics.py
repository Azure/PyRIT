# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
from unittest.mock import patch

from pyrit.identifiers import ComponentIdentifier
from pyrit.score import (
    HarmScorerMetrics,
    ObjectiveScorerMetrics,
    ScorerMetricsWithIdentity,
)
from pyrit.score.scorer_evaluation.scorer_metrics_io import (
    _build_eval_dict,
    compute_eval_hash,
    get_all_harm_metrics,
    get_all_objective_metrics,
    replace_evaluation_results,
)


class TestScorerMetricsSerialization:
    """Tests for ScorerMetrics JSON serialization."""

    def test_harm_metrics_to_json_and_from_json(self, tmp_path):
        metrics = HarmScorerMetrics(
            num_responses=10,
            num_human_raters=3,
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

    def test_objective_metrics_to_json_and_from_json(self, tmp_path):
        metrics = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=3,
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


class TestScorerMetricsWithIdentity:
    """Tests for ScorerMetricsWithIdentity dataclass."""

    def test_creation_with_objective_metrics(self):
        scorer_id = ComponentIdentifier(
            class_name="TestScorer",
            class_module="test.module",
        )
        metrics = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=2,
            accuracy=0.9,
            accuracy_standard_error=0.05,
            f1_score=0.85,
            precision=0.88,
            recall=0.82,
        )

        result = ScorerMetricsWithIdentity(
            scorer_identifier=scorer_id,
            metrics=metrics,
        )

        assert result.scorer_identifier.class_name == "TestScorer"
        assert result.metrics.accuracy == 0.9
        assert result.metrics.f1_score == 0.85

    def test_creation_with_harm_metrics(self):
        scorer_id = ComponentIdentifier(
            class_name="HarmScorer",
            class_module="test.module",
        )
        metrics = HarmScorerMetrics(
            num_responses=20,
            num_human_raters=3,
            mean_absolute_error=0.15,
            mae_standard_error=0.02,
            t_statistic=2.5,
            p_value=0.02,
            krippendorff_alpha_combined=0.75,
            harm_category="hate_speech",
        )

        result = ScorerMetricsWithIdentity(
            scorer_identifier=scorer_id,
            metrics=metrics,
        )

        assert result.scorer_identifier.class_name == "HarmScorer"
        assert result.metrics.mean_absolute_error == 0.15
        assert result.metrics.harm_category == "hate_speech"

    def test_repr(self):
        scorer_id = ComponentIdentifier(
            class_name="MyScorer",
            class_module="test.module",
        )
        metrics = ObjectiveScorerMetrics(
            num_responses=5,
            num_human_raters=1,
            accuracy=0.8,
            accuracy_standard_error=0.1,
            f1_score=0.75,
            precision=0.8,
            recall=0.7,
        )

        result = ScorerMetricsWithIdentity(
            scorer_identifier=scorer_id,
            metrics=metrics,
        )

        repr_str = repr(result)
        assert "MyScorer" in repr_str
        assert "ObjectiveScorerMetrics" in repr_str


class TestGetAllObjectiveMetrics:
    """Tests for the get_all_objective_metrics function."""

    def _create_objective_jsonl(self, path: Path) -> None:
        """Helper to create a test objective metrics JSONL file."""
        entries = [
            {
                "class_name": "SelfAskRefusalScorer",
                "class_module": "pyrit.score",
                "class_description": "Refusal scorer",
                "identifier_type": "instance",
                "system_prompt_template": "test prompt",
                "hash": "abc123",
                "metrics": {
                    "num_responses": 100,
                    "num_human_raters": 2,
                    "accuracy": 0.92,
                    "accuracy_standard_error": 0.03,
                    "f1_score": 0.88,
                    "precision": 0.90,
                    "recall": 0.86,
                    "num_scorer_trials": 3,
                },
            },
            {
                "class_name": "SelfAskTrueFalseScorer",
                "class_module": "pyrit.score",
                "class_description": "True/False scorer",
                "identifier_type": "instance",
                "system_prompt_template": "another prompt",
                "hash": "def456",
                "metrics": {
                    "num_responses": 50,
                    "num_human_raters": 3,
                    "accuracy": 0.85,
                    "accuracy_standard_error": 0.05,
                    "f1_score": 0.80,
                    "precision": 0.82,
                    "recall": 0.78,
                    "num_scorer_trials": 1,
                },
            },
        ]
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_get_all_objective_metrics_loads_correctly(self, tmp_path):
        """Test loading objective metrics."""
        objective_dir = tmp_path / "objective"
        objective_dir.mkdir(parents=True, exist_ok=True)
        objective_file = objective_dir / "objective_achieved_metrics.jsonl"
        self._create_objective_jsonl(objective_file)

        with patch("pyrit.score.scorer_evaluation.scorer_metrics_io.SCORER_EVALS_PATH", tmp_path):
            results = get_all_objective_metrics()

        assert len(results) == 2
        assert all(isinstance(r.metrics, ObjectiveScorerMetrics) for r in results)

        # Check first entry
        first = results[0]
        assert first.scorer_identifier.class_name == "SelfAskRefusalScorer"
        assert first.metrics.accuracy == 0.92
        assert first.metrics.f1_score == 0.88

    def test_get_all_objective_metrics_custom_file_path(self, tmp_path):
        """Test loading from a custom file path."""
        custom_file = tmp_path / "custom_results.jsonl"
        self._create_objective_jsonl(custom_file)

        results = get_all_objective_metrics(file_path=custom_file)

        assert len(results) == 2
        assert all(isinstance(r.metrics, ObjectiveScorerMetrics) for r in results)

    def test_get_all_objective_metrics_empty_file(self, tmp_path):
        """Test handling of empty JSONL file."""
        objective_dir = tmp_path / "objective"
        objective_dir.mkdir(parents=True, exist_ok=True)
        empty_file = objective_dir / "objective_achieved_metrics.jsonl"
        empty_file.touch()

        with patch("pyrit.score.scorer_evaluation.scorer_metrics_io.SCORER_EVALS_PATH", tmp_path):
            results = get_all_objective_metrics()

        assert results == []

    def test_get_all_objective_metrics_missing_file(self, tmp_path):
        """Test handling of missing file (returns empty list)."""
        with patch("pyrit.score.scorer_evaluation.scorer_metrics_io.SCORER_EVALS_PATH", tmp_path):
            results = get_all_objective_metrics()

        assert results == []

    def test_get_all_objective_metrics_malformed_entry_skipped(self, tmp_path):
        """Test that malformed entries are skipped with warning."""
        objective_dir = tmp_path / "objective"
        objective_dir.mkdir(parents=True, exist_ok=True)
        file_path = objective_dir / "objective_achieved_metrics.jsonl"

        # Write one valid and one malformed entry
        with open(file_path, "w") as f:
            # Valid entry
            valid = {
                "class_name": "ValidScorer",
                "class_module": "test.module",
                "class_description": "A valid scorer for testing",
                "identifier_type": "instance",
                "metrics": {
                    "num_responses": 10,
                    "num_human_raters": 1,
                    "accuracy": 0.9,
                    "accuracy_standard_error": 0.05,
                    "f1_score": 0.85,
                    "precision": 0.88,
                    "recall": 0.82,
                },
            }
            f.write(json.dumps(valid) + "\n")
            # Malformed entry (missing required fields)
            malformed = {"class_name": "BadScorer", "metrics": {"accuracy": 0.5}}
            f.write(json.dumps(malformed) + "\n")

        with patch("pyrit.score.scorer_evaluation.scorer_metrics_io.SCORER_EVALS_PATH", tmp_path):
            results = get_all_objective_metrics()

        # Only the valid entry should be loaded
        assert len(results) == 1
        assert results[0].scorer_identifier.class_name == "ValidScorer"

    def test_get_all_objective_metrics_sortable_by_metric(self, tmp_path):
        """Test that results can be sorted by metrics attributes."""
        objective_dir = tmp_path / "objective"
        objective_dir.mkdir(parents=True, exist_ok=True)
        objective_file = objective_dir / "objective_achieved_metrics.jsonl"
        self._create_objective_jsonl(objective_file)

        with patch("pyrit.score.scorer_evaluation.scorer_metrics_io.SCORER_EVALS_PATH", tmp_path):
            results = get_all_objective_metrics()

        # Sort by f1_score descending
        sorted_results = sorted(results, key=lambda x: x.metrics.f1_score, reverse=True)

        assert sorted_results[0].metrics.f1_score == 0.88
        assert sorted_results[1].metrics.f1_score == 0.80

    def test_get_all_objective_metrics_scorer_identifier_reconstructed(self, tmp_path):
        """Test that ComponentIdentifier is properly reconstructed with all fields."""
        objective_dir = tmp_path / "objective"
        objective_dir.mkdir(parents=True, exist_ok=True)
        file_path = objective_dir / "objective_achieved_metrics.jsonl"

        entry = {
            "class_name": "ComplexScorer",
            "class_module": "pyrit.score.complex_scorer",
            "class_description": "A complex scorer for testing",
            "identifier_type": "instance",
            "system_prompt_template": "sha256:abcd1234",
            "user_prompt_template": "user template",
            "target_info": {"model": "gpt-4", "temperature": 0.7},
            "scorer_specific_params": {"threshold": 0.5},
            "hash": "fullhash123",
            "metrics": {
                "num_responses": 10,
                "num_human_raters": 1,
                "accuracy": 0.9,
                "accuracy_standard_error": 0.05,
                "f1_score": 0.85,
                "precision": 0.88,
                "recall": 0.82,
            },
        }
        with open(file_path, "w") as f:
            f.write(json.dumps(entry) + "\n")

        with patch("pyrit.score.scorer_evaluation.scorer_metrics_io.SCORER_EVALS_PATH", tmp_path):
            results = get_all_objective_metrics()

        assert len(results) == 1
        scorer_id = results[0].scorer_identifier

        assert scorer_id.class_name == "ComplexScorer"
        assert scorer_id.params["system_prompt_template"] == "sha256:abcd1234"
        assert scorer_id.params["user_prompt_template"] == "user template"
        assert scorer_id.params["target_info"] == {"model": "gpt-4", "temperature": 0.7}
        assert scorer_id.params["scorer_specific_params"] == {"threshold": 0.5}


class TestGetAllHarmMetrics:
    """Tests for the get_all_harm_metrics function."""

    def _create_harm_jsonl(self, path: Path) -> None:
        """Helper to create a test harm metrics JSONL file."""
        entries = [
            {
                "class_name": "SelfAskLikertScorer",
                "class_module": "pyrit.score",
                "class_description": "Likert scorer",
                "identifier_type": "instance",
                "system_prompt_template": "likert prompt",
                "hash": "harm123",
                "harm_category": "hate_speech",
                "metrics": {
                    "num_responses": 75,
                    "num_human_raters": 4,
                    "mean_absolute_error": 0.12,
                    "mae_standard_error": 0.02,
                    "t_statistic": 1.5,
                    "p_value": 0.14,
                    "krippendorff_alpha_combined": 0.78,
                    "harm_category": "hate_speech",
                    "num_scorer_trials": 2,
                },
            },
        ]
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_get_all_harm_metrics_loads_correctly(self, tmp_path):
        """Test loading harm metrics."""
        harm_path = tmp_path / "harm"
        harm_path.mkdir()
        harm_file = harm_path / "hate_speech_metrics.jsonl"
        self._create_harm_jsonl(harm_file)

        with patch("pyrit.score.scorer_evaluation.scorer_metrics_io.SCORER_EVALS_PATH", tmp_path):
            results = get_all_harm_metrics("hate_speech")

        assert len(results) == 1
        assert isinstance(results[0].metrics, HarmScorerMetrics)
        assert results[0].scorer_identifier.class_name == "SelfAskLikertScorer"
        assert results[0].metrics.mean_absolute_error == 0.12
        assert results[0].metrics.harm_category == "hate_speech"

    def test_get_all_harm_metrics_different_category(self, tmp_path):
        """Test loading from a different harm category."""
        harm_path = tmp_path / "harm"
        harm_path.mkdir()
        violence_file = harm_path / "violence_metrics.jsonl"
        self._create_harm_jsonl(violence_file)

        with patch("pyrit.score.scorer_evaluation.scorer_metrics_io.SCORER_EVALS_PATH", tmp_path):
            results = get_all_harm_metrics("violence")

        assert len(results) == 1
        assert isinstance(results[0].metrics, HarmScorerMetrics)

    def test_get_all_harm_metrics_empty_file(self, tmp_path):
        """Test handling of empty JSONL file."""
        harm_path = tmp_path / "harm"
        harm_path.mkdir()
        empty_file = harm_path / "hate_speech_metrics.jsonl"
        empty_file.touch()

        with patch("pyrit.score.scorer_evaluation.scorer_metrics_io.SCORER_EVALS_PATH", tmp_path):
            results = get_all_harm_metrics("hate_speech")

        assert results == []

    def test_get_all_harm_metrics_missing_file(self, tmp_path):
        """Test handling of missing file (returns empty list)."""
        with patch("pyrit.score.scorer_evaluation.scorer_metrics_io.SCORER_EVALS_PATH", tmp_path):
            results = get_all_harm_metrics("nonexistent_category")

        assert results == []


class TestReplaceEvaluationResults:
    """Tests for replace_evaluation_results function."""

    def test_replace_adds_new_entry(self, tmp_path):
        """Test that replace_evaluation_results adds a new entry when none exists."""
        result_file = tmp_path / "test_results.jsonl"

        scorer_identifier = ComponentIdentifier(
            class_name="TestScorer",
            class_module="test.module",
        )

        metrics = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=3,
            accuracy=0.9,
            accuracy_standard_error=0.05,
            f1_score=0.8,
            precision=0.85,
            recall=0.75,
            num_scorer_trials=3,
            dataset_version="1.0",
        )

        replace_evaluation_results(
            file_path=result_file,
            scorer_identifier=scorer_identifier,
            metrics=metrics,
        )

        # Verify the file contains the entry
        with open(result_file) as f:
            lines = f.readlines()

        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["hash"] == scorer_identifier.hash
        assert entry["metrics"]["accuracy"] == 0.9

    def test_replace_replaces_existing_entry(self, tmp_path):
        """Test that replace_evaluation_results replaces existing entry with same hash."""
        result_file = tmp_path / "test_results.jsonl"

        scorer_identifier = ComponentIdentifier(
            class_name="TestScorer",
            class_module="test.module",
        )

        # Add initial entry
        initial_metrics = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=3,
            accuracy=0.7,
            accuracy_standard_error=0.05,
            f1_score=0.6,
            precision=0.65,
            recall=0.55,
            num_scorer_trials=1,
            dataset_version="1.0",
        )

        replace_evaluation_results(
            file_path=result_file,
            scorer_identifier=scorer_identifier,
            metrics=initial_metrics,
        )

        # Replace with updated metrics
        updated_metrics = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=3,
            accuracy=0.9,
            accuracy_standard_error=0.02,
            f1_score=0.85,
            precision=0.88,
            recall=0.82,
            num_scorer_trials=5,
            dataset_version="1.0",
        )

        replace_evaluation_results(
            file_path=result_file,
            scorer_identifier=scorer_identifier,
            metrics=updated_metrics,
        )

        # Verify only one entry exists with updated values
        with open(result_file) as f:
            lines = f.readlines()

        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["hash"] == scorer_identifier.hash
        assert entry["metrics"]["accuracy"] == 0.9
        assert entry["metrics"]["num_scorer_trials"] == 5

    def test_replace_preserves_other_entries(self, tmp_path):
        """Test that replace_evaluation_results preserves entries with different hashes."""
        result_file = tmp_path / "test_results.jsonl"

        # Add first scorer
        scorer1 = ComponentIdentifier(
            class_name="TestScorer1",
            class_module="test.module",
        )
        metrics1 = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=3,
            accuracy=0.8,
            accuracy_standard_error=0.05,
            f1_score=0.75,
            precision=0.78,
            recall=0.72,
            num_scorer_trials=3,
            dataset_version="1.0",
        )
        replace_evaluation_results(
            file_path=result_file,
            scorer_identifier=scorer1,
            metrics=metrics1,
        )

        # Add second scorer
        scorer2 = ComponentIdentifier(
            class_name="TestScorer2",
            class_module="test.module",
        )
        metrics2 = ObjectiveScorerMetrics(
            num_responses=15,
            num_human_raters=4,
            accuracy=0.85,
            accuracy_standard_error=0.03,
            f1_score=0.8,
            precision=0.82,
            recall=0.78,
            num_scorer_trials=3,
            dataset_version="1.0",
        )
        replace_evaluation_results(
            file_path=result_file,
            scorer_identifier=scorer2,
            metrics=metrics2,
        )

        # Now replace scorer1 with updated metrics
        updated_metrics1 = ObjectiveScorerMetrics(
            num_responses=10,
            num_human_raters=3,
            accuracy=0.95,
            accuracy_standard_error=0.02,
            f1_score=0.9,
            precision=0.92,
            recall=0.88,
            num_scorer_trials=5,
            dataset_version="2.0",
        )
        replace_evaluation_results(
            file_path=result_file,
            scorer_identifier=scorer1,
            metrics=updated_metrics1,
        )

        # Verify both entries exist, scorer1 updated, scorer2 preserved
        with open(result_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        entries = [json.loads(line) for line in lines]
        hashes = {e["hash"]: e for e in entries}

        assert scorer1.hash in hashes
        assert scorer2.hash in hashes
        assert hashes[scorer1.hash]["metrics"]["accuracy"] == 0.95
        assert hashes[scorer2.hash]["metrics"]["accuracy"] == 0.85


class TestBuildEvalDict:
    """Tests for the _build_eval_dict function."""

    def test_basic_identifier_without_params_or_children(self):
        """Test _build_eval_dict with a simple identifier with no params or children."""
        identifier = ComponentIdentifier(
            class_name="SimpleScorer",
            class_module="pyrit.score",
        )
        result = _build_eval_dict(identifier)

        assert result["class_name"] == "SimpleScorer"
        assert result["class_module"] == "pyrit.score"
        assert "children" not in result

    def test_includes_all_non_none_params(self):
        """Test that all non-None params are included in the eval dict."""
        identifier = ComponentIdentifier(
            class_name="ParamScorer",
            class_module="pyrit.score",
            params={"threshold": 0.5, "template": "prompt_text", "mode": "strict"},
        )
        result = _build_eval_dict(identifier)

        assert result["threshold"] == 0.5
        assert result["template"] == "prompt_text"
        assert result["mode"] == "strict"

    def test_param_allowlist_filters_params(self):
        """Test that param_allowlist restricts which params are included."""
        identifier = ComponentIdentifier(
            class_name="FilteredScorer",
            class_module="pyrit.score",
            params={"threshold": 0.5, "template": "prompt_text", "mode": "strict"},
        )
        result = _build_eval_dict(identifier, param_allowlist=frozenset({"threshold", "mode"}))

        assert result["threshold"] == 0.5
        assert result["mode"] == "strict"
        assert "template" not in result

    def test_none_params_are_excluded(self):
        """Test that None-valued params are excluded from the eval dict."""
        identifier = ComponentIdentifier(
            class_name="NoneScorer",
            class_module="pyrit.score",
            params={"threshold": 0.5, "optional_field": None},
        )
        # Note: ComponentIdentifier filters None in .of(), but direct construction allows it
        result = _build_eval_dict(identifier)

        assert result["threshold"] == 0.5
        assert "optional_field" not in result

    def test_children_hashed_with_behavioral_params_only(self):
        """Test that target children are projected to behavioral params only."""
        child = ComponentIdentifier(
            class_name="ChildTarget",
            class_module="pyrit.target",
            params={
                "model_name": "gpt-4",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_requests_per_minute": 100,
                "endpoint": "https://example.com",
            },
        )
        identifier = ComponentIdentifier(
            class_name="ParentScorer",
            class_module="pyrit.score",
            children={"prompt_target": child},
        )
        result = _build_eval_dict(identifier)

        assert "children" in result
        # The child hash should be a string (hashed), not the full child dict
        assert isinstance(result["children"]["prompt_target"], str)

    def test_children_with_different_operational_params_produce_same_hash(self):
        """Test that target children differing only in operational params produce the same child hash."""
        child1 = ComponentIdentifier(
            class_name="ChildTarget",
            class_module="pyrit.target",
            params={
                "model_name": "gpt-4",
                "temperature": 0.7,
                "endpoint": "https://endpoint-a.com",
                "max_requests_per_minute": 50,
            },
        )
        child2 = ComponentIdentifier(
            class_name="ChildTarget",
            class_module="pyrit.target",
            params={
                "model_name": "gpt-4",
                "temperature": 0.7,
                "endpoint": "https://endpoint-b.com",
                "max_requests_per_minute": 200,
            },
        )
        id1 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child1},
        )
        id2 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child2},
        )
        result1 = _build_eval_dict(id1)
        result2 = _build_eval_dict(id2)

        assert result1["children"]["prompt_target"] == result2["children"]["prompt_target"]

    def test_children_with_different_behavioral_params_produce_different_hash(self):
        """Test that target children differing in behavioral params produce different child hashes."""
        child1 = ComponentIdentifier(
            class_name="ChildTarget",
            class_module="pyrit.target",
            params={"model_name": "gpt-4", "temperature": 0.7},
        )
        child2 = ComponentIdentifier(
            class_name="ChildTarget",
            class_module="pyrit.target",
            params={"model_name": "gpt-3.5-turbo", "temperature": 0.7},
        )
        id1 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child1},
        )
        id2 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child2},
        )
        result1 = _build_eval_dict(id1)
        result2 = _build_eval_dict(id2)

        assert result1["children"]["prompt_target"] != result2["children"]["prompt_target"]

    def test_multiple_children_as_list(self):
        """Test that list-valued children produce a list of hashes."""
        child_a = ComponentIdentifier(
            class_name="ChildA",
            class_module="pyrit.target",
            params={"model_name": "gpt-4"},
        )
        child_b = ComponentIdentifier(
            class_name="ChildB",
            class_module="pyrit.target",
            params={"model_name": "gpt-3.5-turbo"},
        )
        identifier = ComponentIdentifier(
            class_name="MultiChildScorer",
            class_module="pyrit.score",
            children={"targets": [child_a, child_b]},
        )
        result = _build_eval_dict(identifier)

        assert "children" in result
        assert isinstance(result["children"]["targets"], list)
        assert len(result["children"]["targets"]) == 2

    def test_single_child_list_unwrapped(self):
        """Test that a single-element child list is unwrapped to a scalar hash."""
        child = ComponentIdentifier(
            class_name="OnlyChild",
            class_module="pyrit.target",
            params={"model_name": "gpt-4"},
        )
        identifier = ComponentIdentifier(
            class_name="SingleChildScorer",
            class_module="pyrit.score",
            children={"target": child},
        )
        result = _build_eval_dict(identifier)

        # Single child should be a scalar string, not a list
        assert isinstance(result["children"]["target"], str)

    def test_no_children_key_when_empty(self):
        """Test that 'children' key is absent when there are no children."""
        identifier = ComponentIdentifier(
            class_name="NoChildScorer",
            class_module="pyrit.score",
            params={"threshold": 0.5},
        )
        result = _build_eval_dict(identifier)

        assert "children" not in result

    def test_non_target_children_include_all_params(self):
        """Test that non-target children (e.g., sub-scorers) include all params, not just behavioral ones."""
        child = ComponentIdentifier(
            class_name="SubScorer",
            class_module="pyrit.score",
            params={
                "model_name": "gpt-4",
                "temperature": 0.7,
                "system_prompt_template": "custom_prompt",
                "threshold": 0.8,
            },
        )
        identifier = ComponentIdentifier(
            class_name="ParentScorer",
            class_module="pyrit.score",
            children={"sub_scorer": child},
        )
        result = _build_eval_dict(identifier)

        assert "children" in result
        assert isinstance(result["children"]["sub_scorer"], str)

    def test_non_target_children_with_different_params_produce_different_hash(self):
        """Test that non-target children differing in any param produce different hashes."""
        child1 = ComponentIdentifier(
            class_name="SubScorer",
            class_module="pyrit.score",
            params={"system_prompt_template": "prompt_a", "endpoint": "https://a.com"},
        )
        child2 = ComponentIdentifier(
            class_name="SubScorer",
            class_module="pyrit.score",
            params={"system_prompt_template": "prompt_a", "endpoint": "https://b.com"},
        )
        id1 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"sub_scorer": child1},
        )
        id2 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"sub_scorer": child2},
        )
        result1 = _build_eval_dict(id1)
        result2 = _build_eval_dict(id2)

        # Non-target children use full eval treatment, so all params matter
        assert result1["children"]["sub_scorer"] != result2["children"]["sub_scorer"]

    def test_target_vs_non_target_children_handled_differently(self):
        """Test that target children filter params while non-target children keep all params."""
        child = ComponentIdentifier(
            class_name="SomeComponent",
            class_module="pyrit.target",
            params={
                "model_name": "gpt-4",
                "endpoint": "https://example.com",
            },
        )

        # Same child as a target child (behavioral filtering applies)
        id_as_target = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child},
        )
        # Same child as a non-target child (full eval treatment)
        id_as_non_target = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"sub_scorer": child},
        )

        result_target = _build_eval_dict(id_as_target)
        result_non_target = _build_eval_dict(id_as_non_target)

        # The child hashes should differ because target filtering drops "endpoint"
        assert result_target["children"]["prompt_target"] != result_non_target["children"]["sub_scorer"]

    def test_converter_target_children_filtered_like_prompt_target(self):
        """Test that converter_target children are also filtered to behavioral params only."""
        child1 = ComponentIdentifier(
            class_name="ConverterTarget",
            class_module="pyrit.target",
            params={
                "model_name": "gpt-4",
                "temperature": 0.7,
                "endpoint": "https://endpoint-a.com",
            },
        )
        child2 = ComponentIdentifier(
            class_name="ConverterTarget",
            class_module="pyrit.target",
            params={
                "model_name": "gpt-4",
                "temperature": 0.7,
                "endpoint": "https://endpoint-b.com",
            },
        )
        id1 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"converter_target": child1},
        )
        id2 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"converter_target": child2},
        )
        result1 = _build_eval_dict(id1)
        result2 = _build_eval_dict(id2)

        # Operational param "endpoint" should be filtered, so hashes match
        assert result1["children"]["converter_target"] == result2["children"]["converter_target"]


class TestComputeEvalHash:
    """Tests for the compute_eval_hash function."""

    def test_deterministic_for_same_identifier(self):
        """Test that compute_eval_hash returns the same hash for the same identifier."""
        identifier = ComponentIdentifier(
            class_name="StableScorer",
            class_module="pyrit.score",
            params={"threshold": 0.5},
        )
        hash1 = compute_eval_hash(identifier)
        hash2 = compute_eval_hash(identifier)

        assert hash1 == hash2

    def test_returns_hex_string(self):
        """Test that compute_eval_hash returns a valid hex string."""
        identifier = ComponentIdentifier(
            class_name="HexScorer",
            class_module="pyrit.score",
        )
        result = compute_eval_hash(identifier)

        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in result)

    def test_different_class_names_produce_different_hashes(self):
        """Test that different class names produce different eval hashes."""
        id1 = ComponentIdentifier(class_name="ScorerA", class_module="pyrit.score")
        id2 = ComponentIdentifier(class_name="ScorerB", class_module="pyrit.score")

        assert compute_eval_hash(id1) != compute_eval_hash(id2)

    def test_different_params_produce_different_hashes(self):
        """Test that different params produce different eval hashes."""
        id1 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            params={"threshold": 0.5},
        )
        id2 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            params={"threshold": 0.8},
        )

        assert compute_eval_hash(id1) != compute_eval_hash(id2)

    def test_eval_hash_differs_from_component_hash(self):
        """Test that eval hash differs from the ComponentIdentifier.hash for target children with operational params."""
        child = ComponentIdentifier(
            class_name="Target",
            class_module="pyrit.target",
            params={
                "model_name": "gpt-4",
                "endpoint": "https://example.com",
            },
        )
        identifier = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child},
        )

        eval_hash = compute_eval_hash(identifier)
        component_hash = identifier.hash

        # They should differ because eval hash filters operational params from target children
        assert eval_hash != component_hash

    def test_operational_child_params_ignored_in_eval_hash(self):
        """Test that operational params on target children don't affect eval hash."""
        child1 = ComponentIdentifier(
            class_name="Target",
            class_module="pyrit.target",
            params={
                "model_name": "gpt-4",
                "temperature": 0.7,
                "endpoint": "https://endpoint-a.com",
                "max_requests_per_minute": 50,
            },
        )
        child2 = ComponentIdentifier(
            class_name="Target",
            class_module="pyrit.target",
            params={
                "model_name": "gpt-4",
                "temperature": 0.7,
                "endpoint": "https://endpoint-b.com",
                "max_requests_per_minute": 200,
            },
        )
        id1 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child1},
        )
        id2 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child2},
        )

        assert compute_eval_hash(id1) == compute_eval_hash(id2)

    def test_behavioral_child_params_affect_eval_hash(self):
        """Test that behavioral params on target children do affect eval hash."""
        child1 = ComponentIdentifier(
            class_name="Target",
            class_module="pyrit.target",
            params={"model_name": "gpt-4", "temperature": 0.7},
        )
        child2 = ComponentIdentifier(
            class_name="Target",
            class_module="pyrit.target",
            params={"model_name": "gpt-4", "temperature": 0.0},
        )
        id1 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child1},
        )
        id2 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            children={"prompt_target": child2},
        )

        assert compute_eval_hash(id1) != compute_eval_hash(id2)

    def test_scorer_own_params_all_included(self):
        """Test that all of the scorer's own params (not just behavioral) are included."""
        id1 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            params={"system_prompt_template": "template_a"},
        )
        id2 = ComponentIdentifier(
            class_name="Scorer",
            class_module="pyrit.score",
            params={"system_prompt_template": "template_b"},
        )

        assert compute_eval_hash(id1) != compute_eval_hash(id2)
