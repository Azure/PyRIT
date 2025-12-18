# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from unittest.mock import patch

import pytest

from pyrit.score.scorer_evaluation.scorer_evaluator import (
    HarmScorerMetrics,
    ObjectiveScorerMetrics,
)
from pyrit.score.scorer_evaluation.scorer_metrics_registry import (
    RegistryType,
    ScorerMetricsEntry,
    ScorerMetricsRegistry,
)
from pyrit.score.scorer_identifier import ScorerIdentifier


@pytest.fixture
def temp_registry_files(tmp_path):
    """Create temporary registry files and patch the registry to use them."""
    harm_file = tmp_path / "harm_registry.jsonl"
    objective_file = tmp_path / "objective_registry.jsonl"

    # Create empty files
    harm_file.touch()
    objective_file.touch()

    # Patch the registry file paths
    with patch.dict(
        ScorerMetricsRegistry._REGISTRY_FILES,
        {
            RegistryType.HARM: harm_file,
            RegistryType.OBJECTIVE: objective_file,
        },
    ):
        yield {"harm": harm_file, "objective": objective_file}


@pytest.fixture
def sample_objective_metrics():
    """Sample ObjectiveScorerMetrics for testing."""
    return ObjectiveScorerMetrics(
        accuracy=0.85,
        accuracy_standard_error=0.02,
        f1_score=0.82,
        precision=0.88,
        recall=0.77,
    )


@pytest.fixture
def sample_harm_metrics():
    """Sample HarmScorerMetrics for testing."""
    return HarmScorerMetrics(
        mean_absolute_error=0.15,
        mae_standard_error=0.03,
        t_statistic=2.5,
        p_value=0.02,
        krippendorff_alpha_combined=0.75,
        krippendorff_alpha_humans=0.80,
        krippendorff_alpha_model=0.70,
    )


@pytest.fixture
def sample_scorer_identifier():
    """Sample ScorerIdentifier for testing."""
    return ScorerIdentifier(
        type="TestScorer",
        system_prompt_template="Test system prompt",
        target_info={"model_name": "gpt-4", "temperature": 0.7},
    )


class TestScorerMetricsEntry:
    """Test ScorerMetricsEntry NamedTuple."""

    def test_scorer_metrics_entry_creation(self, sample_objective_metrics):
        """Test creating a ScorerMetricsEntry."""
        scorer_id = {"__type__": "TestScorer"}
        entry = ScorerMetricsEntry(
            scorer_identifier=scorer_id,
            metrics=sample_objective_metrics,
        )

        assert entry.scorer_identifier == scorer_id
        assert entry.metrics == sample_objective_metrics

    def test_scorer_metrics_entry_unpacking(self, sample_objective_metrics):
        """Test unpacking a ScorerMetricsEntry."""
        scorer_id = {"__type__": "TestScorer"}
        entry = ScorerMetricsEntry(
            scorer_identifier=scorer_id,
            metrics=sample_objective_metrics,
        )

        unpacked_id, unpacked_metrics = entry
        assert unpacked_id == scorer_id
        assert unpacked_metrics == sample_objective_metrics


class TestScorerMetricsRegistryAddEntry:
    """Test adding entries to the registry."""

    def test_add_objective_entry(self, temp_registry_files, sample_scorer_identifier, sample_objective_metrics):
        """Test adding an objective entry to the registry."""
        # Reset singleton to ensure fresh instance
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        registry.add_entry(
            scorer_identifier=sample_scorer_identifier,
            metrics=sample_objective_metrics,
            registry_type=RegistryType.OBJECTIVE,
            dataset_version="v1.0",
        )

        # Verify entry was written
        with open(temp_registry_files["objective"], "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["__type__"] == "TestScorer"
            assert entry["dataset_version"] == "v1.0"
            assert "hash" in entry
            assert entry["metrics"]["accuracy"] == 0.85

    def test_add_harm_entry(self, temp_registry_files, sample_scorer_identifier, sample_harm_metrics):
        """Test adding a harm entry to the registry."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        registry.add_entry(
            scorer_identifier=sample_scorer_identifier,
            metrics=sample_harm_metrics,
            registry_type=RegistryType.HARM,
            dataset_version="v2.0",
        )

        # Verify entry was written
        with open(temp_registry_files["harm"], "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["__type__"] == "TestScorer"
            assert entry["dataset_version"] == "v2.0"
            assert entry["metrics"]["mean_absolute_error"] == 0.15

    def test_add_entry_long_prompt_hashed(self, temp_registry_files, sample_objective_metrics):
        """Test that long prompts are hashed when stored."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        long_prompt = "X" * 200
        scorer_identifier = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template=long_prompt,
        )

        registry.add_entry(
            scorer_identifier=scorer_identifier,
            metrics=sample_objective_metrics,
            registry_type=RegistryType.OBJECTIVE,
            dataset_version="v1.0",
        )

        # Verify prompt was hashed
        with open(temp_registry_files["objective"], "r") as f:
            entry = json.loads(f.readline())
            assert entry["system_prompt_template"].startswith("sha256:")

    def test_add_multiple_entries(self, temp_registry_files, sample_scorer_identifier, sample_objective_metrics):
        """Test adding multiple entries."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        for i in range(3):
            scorer_id = ScorerIdentifier(type=f"Scorer{i}")
            registry.add_entry(
                scorer_identifier=scorer_id,
                metrics=sample_objective_metrics,
                registry_type=RegistryType.OBJECTIVE,
                dataset_version="v1.0",
            )

        with open(temp_registry_files["objective"], "r") as f:
            lines = f.readlines()
            assert len(lines) == 3


class TestScorerMetricsRegistryGetEntries:
    """Test retrieving entries from the registry."""

    def test_get_entries_by_hash(self, temp_registry_files, sample_scorer_identifier, sample_objective_metrics):
        """Test retrieving entries by hash."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        # Add entry
        registry.add_entry(
            scorer_identifier=sample_scorer_identifier,
            metrics=sample_objective_metrics,
            registry_type=RegistryType.OBJECTIVE,
            dataset_version="v1.0",
        )

        # Retrieve by hash
        expected_hash = sample_scorer_identifier.compute_hash()
        entries = registry.get_metrics_registry_entries(
            registry_type=RegistryType.OBJECTIVE,
            hash=expected_hash,
        )

        assert len(entries) == 1
        assert entries[0].metrics.accuracy == 0.85

    def test_get_entries_by_type(self, temp_registry_files, sample_objective_metrics):
        """Test retrieving entries by scorer type."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        # Add entries with different types
        for scorer_type in ["ScorerA", "ScorerB", "ScorerA"]:
            scorer_id = ScorerIdentifier(type=scorer_type)
            registry.add_entry(
                scorer_identifier=scorer_id,
                metrics=sample_objective_metrics,
                registry_type=RegistryType.OBJECTIVE,
                dataset_version="v1.0",
            )

        # Retrieve only ScorerA entries
        entries = registry.get_metrics_registry_entries(
            registry_type=RegistryType.OBJECTIVE,
            type="ScorerA",
        )

        assert len(entries) == 2
        for entry in entries:
            assert entry.scorer_identifier["__type__"] == "ScorerA"

    def test_get_entries_by_model_name(self, temp_registry_files, sample_objective_metrics):
        """Test retrieving entries by model name."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        # Add entries with different model names
        for model_name in ["gpt-4", "gpt-3.5", "gpt-4"]:
            scorer_id = ScorerIdentifier(
                type="TestScorer",
                target_info={"model_name": model_name},
            )
            registry.add_entry(
                scorer_identifier=scorer_id,
                metrics=sample_objective_metrics,
                registry_type=RegistryType.OBJECTIVE,
                dataset_version="v1.0",
            )

        entries = registry.get_metrics_registry_entries(
            registry_type=RegistryType.OBJECTIVE,
            model_name="gpt-4",
        )

        assert len(entries) == 2

    def test_get_entries_by_accuracy_threshold(self, temp_registry_files):
        """Test retrieving entries by accuracy threshold."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        # Add entries with different accuracies
        for accuracy in [0.7, 0.8, 0.9]:
            scorer_id = ScorerIdentifier(type=f"Scorer{int(accuracy*10)}")
            metrics = ObjectiveScorerMetrics(
                accuracy=accuracy,
                accuracy_standard_error=0.02,
                f1_score=0.8,
                precision=0.85,
                recall=0.75,
            )
            registry.add_entry(
                scorer_identifier=scorer_id,
                metrics=metrics,
                registry_type=RegistryType.OBJECTIVE,
                dataset_version="v1.0",
            )

        entries = registry.get_metrics_registry_entries(
            registry_type=RegistryType.OBJECTIVE,
            accuracy_threshold=0.75,
        )

        assert len(entries) == 2
        for entry in entries:
            assert entry.metrics.accuracy >= 0.75

    def test_get_entries_sorted_by_accuracy(self, temp_registry_files):
        """Test that objective entries are sorted by accuracy (highest first)."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        # Add entries in random accuracy order
        for accuracy in [0.7, 0.9, 0.8]:
            scorer_id = ScorerIdentifier(type=f"Scorer{int(accuracy*10)}")
            metrics = ObjectiveScorerMetrics(
                accuracy=accuracy,
                accuracy_standard_error=0.02,
                f1_score=0.8,
                precision=0.85,
                recall=0.75,
            )
            registry.add_entry(
                scorer_identifier=scorer_id,
                metrics=metrics,
                registry_type=RegistryType.OBJECTIVE,
                dataset_version="v1.0",
            )

        entries = registry.get_metrics_registry_entries(
            registry_type=RegistryType.OBJECTIVE,
        )

        assert len(entries) == 3
        # Should be sorted highest to lowest accuracy
        assert entries[0].metrics.accuracy == 0.9
        assert entries[1].metrics.accuracy == 0.8
        assert entries[2].metrics.accuracy == 0.7

    def test_get_entries_sorted_by_mae(self, temp_registry_files):
        """Test that harm entries are sorted by MAE (lowest first)."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        # Add entries in random MAE order
        for mae in [0.3, 0.1, 0.2]:
            scorer_id = ScorerIdentifier(type=f"Scorer{int(mae*10)}")
            metrics = HarmScorerMetrics(
                mean_absolute_error=mae,
                mae_standard_error=0.02,
                t_statistic=2.0,
                p_value=0.05,
                krippendorff_alpha_combined=0.8,
                krippendorff_alpha_humans=0.85,
                krippendorff_alpha_model=0.75,
            )
            registry.add_entry(
                scorer_identifier=scorer_id,
                metrics=metrics,
                registry_type=RegistryType.HARM,
                dataset_version="v1.0",
            )

        entries = registry.get_metrics_registry_entries(
            registry_type=RegistryType.HARM,
        )

        assert len(entries) == 3
        # Should be sorted lowest to highest MAE
        assert entries[0].metrics.mean_absolute_error == 0.1
        assert entries[1].metrics.mean_absolute_error == 0.2
        assert entries[2].metrics.mean_absolute_error == 0.3

    def test_get_entries_empty_registry(self, temp_registry_files):
        """Test retrieving from an empty registry."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        entries = registry.get_metrics_registry_entries(
            registry_type=RegistryType.OBJECTIVE,
        )

        assert len(entries) == 0

    def test_get_entries_no_match(self, temp_registry_files, sample_scorer_identifier, sample_objective_metrics):
        """Test retrieving with filters that match nothing."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        registry.add_entry(
            scorer_identifier=sample_scorer_identifier,
            metrics=sample_objective_metrics,
            registry_type=RegistryType.OBJECTIVE,
            dataset_version="v1.0",
        )

        entries = registry.get_metrics_registry_entries(
            registry_type=RegistryType.OBJECTIVE,
            type="NonExistentScorer",
        )

        assert len(entries) == 0


class TestScorerMetricsRegistryByIdentifier:
    """Test get_scorer_registry_metrics_by_identifier method."""

    def test_get_metrics_by_identifier_found(
        self, temp_registry_files, sample_scorer_identifier, sample_objective_metrics
    ):
        """Test retrieving metrics by scorer identifier when entry exists."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        registry.add_entry(
            scorer_identifier=sample_scorer_identifier,
            metrics=sample_objective_metrics,
            registry_type=RegistryType.OBJECTIVE,
            dataset_version="v1.0",
        )

        result = registry.get_scorer_registry_metrics_by_identifier(
            sample_scorer_identifier,
            registry_type=RegistryType.OBJECTIVE,
        )

        assert result is not None
        assert result.accuracy == 0.85

    def test_get_metrics_by_identifier_not_found(self, temp_registry_files, sample_scorer_identifier):
        """Test retrieving metrics by scorer identifier when entry doesn't exist."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        result = registry.get_scorer_registry_metrics_by_identifier(
            sample_scorer_identifier,
            registry_type=RegistryType.OBJECTIVE,
        )

        assert result is None


class TestScorerMetricsRegistryHashConsistency:
    """Test hash consistency between storage and retrieval."""

    def test_hash_consistent_round_trip(self, temp_registry_files, sample_objective_metrics):
        """Test that hash matches when storing and retrieving."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        scorer_identifier = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template="A" * 150,  # Long prompt to test hashing
            target_info={"model_name": "gpt-4"},
        )

        # Store the expected hash before adding
        expected_hash = scorer_identifier.compute_hash()

        # Add entry
        registry.add_entry(
            scorer_identifier=scorer_identifier,
            metrics=sample_objective_metrics,
            registry_type=RegistryType.OBJECTIVE,
            dataset_version="v1.0",
        )

        # Retrieve and verify hash
        entries = registry.get_metrics_registry_entries(
            registry_type=RegistryType.OBJECTIVE,
            hash=expected_hash,
        )

        assert len(entries) == 1
        assert entries[0].scorer_identifier["hash"] == expected_hash

    def test_hash_lookup_works_after_storage(self, temp_registry_files, sample_objective_metrics):
        """Test that we can lookup by hash after storing (simulates the bug fix)."""
        ScorerMetricsRegistry._instances = {}
        registry = ScorerMetricsRegistry()

        # Create identifier with long prompt
        long_prompt = "This is a long prompt " * 20
        scorer_identifier = ScorerIdentifier(
            type="TestScorer",
            system_prompt_template=long_prompt,
        )

        # Add entry
        registry.add_entry(
            scorer_identifier=scorer_identifier,
            metrics=sample_objective_metrics,
            registry_type=RegistryType.OBJECTIVE,
            dataset_version="v1.0",
        )

        # Now lookup using the same identifier
        result = registry.get_scorer_registry_metrics_by_identifier(
            scorer_identifier,
            registry_type=RegistryType.OBJECTIVE,
        )

        # This is the key assertion - the lookup should work
        assert result is not None
        assert result.accuracy == 0.85
