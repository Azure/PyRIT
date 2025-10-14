# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the ScenarioFactory class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyrit.prompt_target import PromptTarget
from pyrit.scenarios import Scenario
from pyrit.setup import ScenarioFactory


@pytest.fixture
def mock_target():
    """Create a mock PromptTarget for testing."""
    return MagicMock(spec=PromptTarget)


@pytest.fixture
def valid_scenario_config():
    """Create a temporary valid scenario configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """
from pathlib import Path

scenario_config = {
    "name": "Test Scenario",
    "description": "A test scenario",
    "attack_runs": [
        {
            "attack_config": Path(__file__).parent / "attack1.py",
            "dataset_config": Path(__file__).parent / "dataset1.py",
        },
        {
            "attack_config": Path(__file__).parent / "attack2.py",
            "dataset_config": Path(__file__).parent / "dataset2.py",
        },
    ],
}
"""
        )
        temp_path = f.name

    yield Path(temp_path)
    Path(temp_path).unlink()


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioFactoryInitialization:
    """Tests for ScenarioFactory initialization and validation."""

    def test_create_scenario_fails_with_nonexistent_config(self, mock_target):
        """Test that creation fails when config file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ScenarioFactory.create_scenario(
                config_path="nonexistent_scenario.py",
                objective_target=mock_target,
            )

    def test_create_scenario_fails_without_scenario_config(self, mock_target):
        """Test that creation fails when config file doesn't define scenario_config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# No scenario_config defined\n")
            temp_path = f.name

        try:
            with pytest.raises(AttributeError, match="must define 'scenario_config' dictionary"):
                ScenarioFactory.create_scenario(
                    config_path=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()

    def test_create_scenario_fails_without_name(self, mock_target):
        """Test that creation fails when scenario_config doesn't include name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
scenario_config = {
    "attack_runs": [],
}
"""
            )
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must include 'name'"):
                ScenarioFactory.create_scenario(
                    config_path=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()

    def test_create_scenario_fails_with_empty_attack_runs(self, mock_target):
        """Test that creation fails when attack_runs list is empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
scenario_config = {
    "name": "Test Scenario",
    "attack_runs": [],
}
"""
            )
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must include 'attack_runs' list"):
                ScenarioFactory.create_scenario(
                    config_path=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()

    def test_create_scenario_fails_with_missing_attack_config(self, mock_target):
        """Test that creation fails when attack run is missing attack_config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from pathlib import Path
scenario_config = {
    "name": "Test Scenario",
    "attack_runs": [
        {
            "dataset_config": Path(__file__).parent / "dataset.py",
        }
    ],
}
"""
            )
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must include 'attack_config'"):
                ScenarioFactory.create_scenario(
                    config_path=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()

    def test_create_scenario_fails_with_missing_dataset_config(self, mock_target):
        """Test that creation fails when attack run is missing dataset_config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from pathlib import Path
scenario_config = {
    "name": "Test Scenario",
    "attack_runs": [
        {
            "attack_config": Path(__file__).parent / "attack.py",
        }
    ],
}
"""
            )
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must include 'dataset_config'"):
                ScenarioFactory.create_scenario(
                    config_path=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioFactoryCreation:
    """Tests for ScenarioFactory scenario creation."""

    @patch("pyrit.scenarios.attack_run.AttackRun.__init__", return_value=None)
    def test_create_scenario_with_valid_config(self, mock_attack_run_init, mock_target, valid_scenario_config):
        """Test successful scenario creation with valid configuration."""
        with (
            patch("pyrit.scenarios.attack_run.AttackRun._validate_config_paths"),
            patch("pyrit.scenarios.attack_run.AttackRun._create_attack"),
            patch("pyrit.scenarios.attack_run.AttackRun._load_dataset"),
        ):
            scenario = ScenarioFactory.create_scenario(
                config_path=valid_scenario_config,
                objective_target=mock_target,
            )

            assert isinstance(scenario, Scenario)
            assert scenario.name == "Test Scenario"
            # Two attack runs should have been created
            assert mock_attack_run_init.call_count == 2

    @patch("pyrit.scenarios.attack_run.AttackRun.__init__", return_value=None)
    def test_create_scenario_passes_memory_labels(self, mock_attack_run_init, mock_target, valid_scenario_config):
        """Test that memory labels are passed to attack runs."""
        memory_labels = {"test": "factory", "category": "scenario"}

        with (
            patch("pyrit.scenarios.attack_run.AttackRun._validate_config_paths"),
            patch("pyrit.scenarios.attack_run.AttackRun._create_attack"),
            patch("pyrit.scenarios.attack_run.AttackRun._load_dataset"),
        ):

            # Verify memory_labels were passed to AttackRun.__init__
            for call in mock_attack_run_init.call_args_list:
                assert call.kwargs["memory_labels"] == memory_labels

    @patch("pyrit.scenarios.attack_run.AttackRun.__init__", return_value=None)
    def test_create_scenario_passes_attack_run_params(self, mock_attack_run_init, mock_target, valid_scenario_config):
        """Test that additional attack run parameters are passed."""
        with (
            patch("pyrit.scenarios.attack_run.AttackRun._validate_config_paths"),
            patch("pyrit.scenarios.attack_run.AttackRun._create_attack"),
            patch("pyrit.scenarios.attack_run.AttackRun._load_dataset"),
        ):

            # Verify custom parameters were passed to AttackRun.__init__
            for call in mock_attack_run_init.call_args_list:
                assert call.kwargs["custom_param"] == "test_value"
                assert call.kwargs["max_retries"] == 5

    @patch("pyrit.scenarios.attack_run.AttackRun.__init__", return_value=None)
    def test_create_scenario_with_run_specific_params(self, mock_attack_run_init, mock_target):
        """Test that run-specific parameters from config override global params."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
from pathlib import Path

scenario_config = {
    "name": "Test Scenario",
    "attack_runs": [
        {
            "attack_config": Path(__file__).parent / "attack1.py",
            "dataset_config": Path(__file__).parent / "dataset1.py",
            "custom_param": "run1_value",
        },
        {
            "attack_config": Path(__file__).parent / "attack2.py",
            "dataset_config": Path(__file__).parent / "dataset2.py",
            "custom_param": "run2_value",
        },
    ],
}
"""
            )
            temp_path = f.name

        try:
            with (
                patch("pyrit.scenarios.attack_run.AttackRun._validate_config_paths"),
                patch("pyrit.scenarios.attack_run.AttackRun._create_attack"),
                patch("pyrit.scenarios.attack_run.AttackRun._load_dataset"),
            ):
                scenario = ScenarioFactory.create_scenario(
                    config_path=temp_path,
                    objective_target=mock_target,
                )

                # Verify scenario was created
                assert isinstance(scenario, Scenario)
                # First run should have run1_value (overrides global)
                assert mock_attack_run_init.call_args_list[0].kwargs["custom_param"] == "run1_value"
                # Second run should have run2_value (overrides global)
                assert mock_attack_run_init.call_args_list[1].kwargs["custom_param"] == "run2_value"
        finally:
            Path(temp_path).unlink()
