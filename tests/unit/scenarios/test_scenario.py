# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the scenarios.Scenario class."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import AttackExecutor
from pyrit.models import AttackOutcome, AttackResult
from pyrit.prompt_target import PromptTarget
from pyrit.scenarios import Scenario
from pyrit.setup import ConfigurationPaths


@pytest.fixture
def mock_target():
    """Create a mock PromptTarget for testing."""
    return MagicMock(spec=PromptTarget)


@pytest.fixture
def valid_attack_config():
    """Create a temporary valid attack configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """
from pyrit.executor.attack import AttackConverterConfig
from pyrit.prompt_converter import AsciiArtConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration

converters = PromptConverterConfiguration.from_converters(converters=[AsciiArtConverter()])
attack_converter_config = AttackConverterConfig(request_converters=converters)

attack_config = {
    "attack_type": "PromptSendingAttack",
    "attack_converter_config": attack_converter_config,
}
"""
        )
        temp_path = f.name

    yield Path(temp_path)
    Path(temp_path).unlink()


@pytest.fixture
def valid_dataset_config():
    """Create a temporary valid dataset configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """
dataset_config = {
    "objectives": ["objective1", "objective2", "objective3"],
}
"""
        )
        temp_path = f.name

    yield Path(temp_path)
    Path(temp_path).unlink()


@pytest.fixture
def sample_attack_results():
    """Create sample attack results for testing."""
    return [
        AttackResult(
            conversation_id="conv-1",
            objective="objective1",
            attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "1"},
            outcome=AttackOutcome.SUCCESS,
            executed_turns=1,
        ),
        AttackResult(
            conversation_id="conv-2",
            objective="objective2",
            attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "2"},
            outcome=AttackOutcome.SUCCESS,
            executed_turns=1,
        ),
        AttackResult(
            conversation_id="conv-3",
            objective="objective3",
            attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "3"},
            outcome=AttackOutcome.FAILURE,
            executed_turns=1,
        ),
    ]


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioInitialization:
    """Tests for Scenario class initialization."""

    def test_init_with_valid_configs(self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config):
        """Test successful initialization with valid configuration files."""
        scenario = Scenario(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        assert scenario._objective_target == mock_target
        assert scenario._attack_config_path == valid_attack_config
        assert scenario._dataset_config_path == valid_dataset_config
        assert scenario._memory_labels == {}
        assert scenario._attack is not None
        assert scenario._dataset_params is not None

    def test_init_with_memory_labels(self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config):
        """Test initialization with memory labels."""
        memory_labels = {"test": "label", "category": "attack"}

        scenario = Scenario(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            memory_labels=memory_labels,
        )

        assert scenario._memory_labels == memory_labels

    def test_init_with_attack_execute_params(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config
    ):
        """Test initialization with additional attack execute parameters."""
        scenario = Scenario(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            max_retries=5,
            custom_param="value",
        )

        assert scenario._attack_execute_params["max_retries"] == 5
        assert scenario._attack_execute_params["custom_param"] == "value"

    def test_init_with_string_paths(self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config):
        """Test initialization with string paths instead of Path objects."""
        scenario = Scenario(
            attack_config=str(valid_attack_config),
            dataset_config=str(valid_dataset_config),
            objective_target=mock_target,
        )

        assert scenario._attack_config_path == valid_attack_config
        assert scenario._dataset_config_path == valid_dataset_config

    def test_init_with_configuration_paths(self, mock_target: PromptTarget):
        """Test initialization with ConfigurationPaths."""
        scenario = Scenario(
            attack_config=ConfigurationPaths.attack.foundry.ascii_art,
            dataset_config=ConfigurationPaths.dataset.harm_bench,
            objective_target=mock_target,
        )

        assert scenario._attack is not None
        assert scenario._dataset_params is not None
        assert "objectives" in scenario._dataset_params


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioValidation:
    """Tests for Scenario validation methods."""

    def test_init_fails_with_nonexistent_attack_config(self, mock_target: PromptTarget, valid_dataset_config):
        """Test that initialization fails when attack config file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Attack configuration file not found"):
            Scenario(
                attack_config="nonexistent_attack.py",
                dataset_config=valid_dataset_config,
                objective_target=mock_target,
            )

    def test_init_fails_with_nonexistent_dataset_config(self, mock_target: PromptTarget, valid_attack_config):
        """Test that initialization fails when dataset config file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Dataset configuration file not found"):
            Scenario(
                attack_config=valid_attack_config,
                dataset_config="nonexistent_dataset.py",
                objective_target=mock_target,
            )

    def test_init_fails_with_directory_as_attack_config(
        self, mock_target: PromptTarget, valid_dataset_config, tmp_path
    ):
        """Test that initialization fails when attack config path is a directory."""
        with pytest.raises(ValueError, match="Attack configuration path is not a file"):
            Scenario(
                attack_config=tmp_path,
                dataset_config=valid_dataset_config,
                objective_target=mock_target,
            )

    def test_init_fails_with_directory_as_dataset_config(
        self, mock_target: PromptTarget, valid_attack_config, tmp_path
    ):
        """Test that initialization fails when dataset config path is a directory."""
        with pytest.raises(ValueError, match="Dataset configuration path is not a file"):
            Scenario(
                attack_config=valid_attack_config,
                dataset_config=tmp_path,
                objective_target=mock_target,
            )

    def test_init_fails_with_invalid_attack_config(self, mock_target: PromptTarget, valid_dataset_config):
        """Test that initialization fails with invalid attack configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Invalid config - no attack_config dictionary\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to create attack from configuration file"):
                Scenario(
                    attack_config=temp_path,
                    dataset_config=valid_dataset_config,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()

    def test_init_fails_with_invalid_dataset_config(self, mock_target: PromptTarget, valid_attack_config):
        """Test that initialization fails with invalid dataset configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Invalid config - no dataset_config dictionary\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to load dataset from configuration file"):
                Scenario(
                    attack_config=valid_attack_config,
                    dataset_config=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioExecution:
    """Tests for Scenario execution methods."""

    @pytest.mark.asyncio
    async def test_run_async_with_valid_scenario(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, sample_attack_results
    ):
        """Test successful execution of a scenario."""
        scenario = Scenario(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        # Mock the executor and attack
        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            results = await scenario.run_async()

            assert len(results) == 3
            assert results == sample_attack_results
            mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async_with_custom_concurrency(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, sample_attack_results
    ):
        """Test execution with custom max_concurrency."""
        scenario = Scenario(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        with patch.object(AttackExecutor, "__init__", return_value=None) as mock_init:
            with patch.object(
                AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = sample_attack_results

                results = await scenario.run_async(max_concurrency=5)

                mock_init.assert_called_once_with(max_concurrency=5)
                assert len(results) == 3

    @pytest.mark.asyncio
    async def test_run_async_passes_memory_labels(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, sample_attack_results
    ):
        """Test that memory labels are passed to the executor."""
        memory_labels = {"test": "scenario", "category": "attack"}

        scenario = Scenario(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            memory_labels=memory_labels,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            await scenario.run_async()

            # Check that memory_labels were passed in the call
            call_kwargs = mock_exec.call_args.kwargs
            assert "memory_labels" in call_kwargs
            assert call_kwargs["memory_labels"] == memory_labels

    @pytest.mark.asyncio
    async def test_run_async_passes_attack_execute_params(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, sample_attack_results
    ):
        """Test that attack execute parameters are passed to the executor."""
        scenario = Scenario(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            custom_param="value",
            max_retries=3,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            await scenario.run_async()

            # Check that custom parameters were passed
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["custom_param"] == "value"
            assert call_kwargs["max_retries"] == 3

    @pytest.mark.asyncio
    async def test_run_async_passes_dataset_params(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, sample_attack_results
    ):
        """Test that dataset parameters are passed to the executor."""
        scenario = Scenario(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            await scenario.run_async()

            # Check that objectives from dataset were passed
            call_kwargs = mock_exec.call_args.kwargs
            assert "objectives" in call_kwargs
            assert call_kwargs["objectives"] == ["objective1", "objective2", "objective3"]

    @pytest.mark.asyncio
    async def test_run_async_handles_execution_failure(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config
    ):
        """Test that execution failures are properly handled and raised."""
        scenario = Scenario(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception("Execution error")

            with pytest.raises(ValueError, match="Failed to execute scenario"):
                await scenario.run_async()


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioIntegration:
    """Integration tests for Scenario with real configuration files."""

    def test_scenario_with_real_configs(self, mock_target: PromptTarget):
        """Test creating a scenario with real configuration files from ConfigurationPaths."""
        scenario = Scenario(
            attack_config=ConfigurationPaths.attack.foundry.ascii_art,
            dataset_config=ConfigurationPaths.dataset.harm_bench,
            objective_target=mock_target,
        )

        assert scenario._attack is not None
        assert scenario._dataset_params is not None
        assert len(scenario._dataset_params["objectives"]) > 0

    @pytest.mark.asyncio
    async def test_full_scenario_execution_flow(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config
    ):
        """Test the complete scenario execution flow end-to-end."""
        memory_labels = {"test": "integration", "scenario": "full"}

        scenario = Scenario(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            memory_labels=memory_labels,
        )

        # Create mock results
        mock_results = [
            AttackResult(
                conversation_id=f"conv-{i}",
                objective=f"objective{i+1}",
                attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": str(i)},
                outcome=AttackOutcome.SUCCESS,
                executed_turns=1,
            )
            for i in range(3)
        ]

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_results

            results = await scenario.run_async(max_concurrency=3)

            # Verify results
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.objective == f"objective{i+1}"
                assert result.outcome == AttackOutcome.SUCCESS

            # Verify the call was made with all expected parameters
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == scenario._attack
            assert call_kwargs["objectives"] == ["objective1", "objective2", "objective3"]
            assert call_kwargs["memory_labels"] == memory_labels
