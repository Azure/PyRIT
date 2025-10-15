# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the scenarios.AttackRun class."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import AttackExecutor
from pyrit.models import AttackOutcome, AttackResult
from pyrit.prompt_converter import Base64Converter, LeetspeakConverter, PromptConverter
from pyrit.prompt_target import PromptTarget
from pyrit.scenarios import AttackRun
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


@pytest.fixture
def valid_converter_config_base64():
    """Create a temporary valid converter configuration file for Base64."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """
from pyrit.prompt_converter import Base64Converter

additional_converter = Base64Converter()
"""
        )
        temp_path = f.name

    yield Path(temp_path)
    Path(temp_path).unlink()


@pytest.fixture
def valid_converter_config_leetspeak():
    """Create a temporary valid converter configuration file for Leetspeak."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """
from pyrit.prompt_converter import LeetspeakConverter

additional_converter = LeetspeakConverter()
"""
        )
        temp_path = f.name

    yield Path(temp_path)
    Path(temp_path).unlink()


@pytest.fixture
def invalid_converter_config_no_attribute():
    """Create an invalid converter configuration file (missing additional_converter)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """
from pyrit.prompt_converter import Base64Converter

# Missing additional_converter attribute
some_other_converter = Base64Converter()
"""
        )
        temp_path = f.name

    yield Path(temp_path)
    Path(temp_path).unlink()


@pytest.fixture
def invalid_converter_config_wrong_type():
    """Create an invalid converter configuration file (wrong type)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """
# additional_converter is not a PromptConverter
additional_converter = "not a converter"
"""
        )
        temp_path = f.name

    yield Path(temp_path)
    Path(temp_path).unlink()


@pytest.mark.usefixtures("patch_central_database")
class TestAttackRunInitialization:
    """Tests for AttackRun class initialization."""

    def test_init_with_valid_configs(self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config):
        """Test successful initialization with valid configuration files."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        assert attack_run._objective_target == mock_target
        assert attack_run._attack_config_path == valid_attack_config
        assert attack_run._dataset_config_path == valid_dataset_config
        assert attack_run._memory_labels == {}
        assert attack_run._attack is not None
        assert attack_run._dataset_params is not None

    def test_init_with_memory_labels(self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config):
        """Test initialization with memory labels."""
        memory_labels = {"test": "label", "category": "attack"}

        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            memory_labels=memory_labels,
        )

        assert attack_run._memory_labels == memory_labels

    def test_init_with_attack_execute_params(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config
    ):
        """Test initialization with additional attack execute parameters."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            max_retries=5,
            custom_param="value",
        )

        assert attack_run._attack_execute_params["max_retries"] == 5
        assert attack_run._attack_execute_params["custom_param"] == "value"

    def test_init_with_string_paths(self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config):
        """Test initialization with string paths instead of Path objects."""
        attack_run = AttackRun(
            attack_config=str(valid_attack_config),
            dataset_config=str(valid_dataset_config),
            objective_target=mock_target,
        )

        assert attack_run._attack_config_path == valid_attack_config
        assert attack_run._dataset_config_path == valid_dataset_config

    def test_init_with_configuration_paths(self, mock_target: PromptTarget):
        """Test initialization with ConfigurationPaths."""
        attack_run = AttackRun(
            attack_config=ConfigurationPaths.attack.foundry.ascii_art,
            dataset_config=ConfigurationPaths.dataset.harm_bench,
            objective_target=mock_target,
        )

        assert attack_run._attack is not None
        assert attack_run._dataset_params is not None
        assert "objectives" in attack_run._dataset_params


@pytest.mark.usefixtures("patch_central_database")
class TestAttackRunValidation:
    """Tests for AttackRun validation methods."""

    def test_init_fails_with_nonexistent_attack_config(self, mock_target: PromptTarget, valid_dataset_config):
        """Test that initialization fails when attack config file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Attack configuration file not found"):
            AttackRun(
                attack_config="nonexistent_attack.py",
                dataset_config=valid_dataset_config,
                objective_target=mock_target,
            )

    def test_init_fails_with_nonexistent_dataset_config(self, mock_target: PromptTarget, valid_attack_config):
        """Test that initialization fails when dataset config file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Dataset configuration file not found"):
            AttackRun(
                attack_config=valid_attack_config,
                dataset_config="nonexistent_dataset.py",
                objective_target=mock_target,
            )

    def test_init_fails_with_directory_as_attack_config(
        self, mock_target: PromptTarget, valid_dataset_config, tmp_path
    ):
        """Test that initialization fails when attack config path is a directory."""
        with pytest.raises(ValueError, match="Attack configuration path is not a file"):
            AttackRun(
                attack_config=tmp_path,
                dataset_config=valid_dataset_config,
                objective_target=mock_target,
            )

    def test_init_fails_with_directory_as_dataset_config(
        self, mock_target: PromptTarget, valid_attack_config, tmp_path
    ):
        """Test that initialization fails when dataset config path is a directory."""
        with pytest.raises(ValueError, match="Dataset configuration path is not a file"):
            AttackRun(
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
                AttackRun(
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
                AttackRun(
                    attack_config=valid_attack_config,
                    dataset_config=temp_path,
                    objective_target=mock_target,
                )
        finally:
            Path(temp_path).unlink()


@pytest.mark.usefixtures("patch_central_database")
class TestAttackRunExecution:
    """Tests for AttackRun execution methods."""

    @pytest.mark.asyncio
    async def test_run_async_with_valid_attack_run(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, sample_attack_results
    ):
        """Test successful execution of an attack run."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        # Mock the executor and attack
        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            results = await attack_run.run_async()

            assert len(results) == 3
            assert results == sample_attack_results
            mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_async_with_custom_concurrency(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, sample_attack_results
    ):
        """Test execution with custom max_concurrency."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        with patch.object(AttackExecutor, "__init__", return_value=None) as mock_init:
            with patch.object(
                AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = sample_attack_results

                results = await attack_run.run_async(max_concurrency=5)

                mock_init.assert_called_once_with(max_concurrency=5)
                assert len(results) == 3

    @pytest.mark.asyncio
    async def test_run_async_passes_memory_labels(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, sample_attack_results
    ):
        """Test that memory labels are passed to the executor."""
        memory_labels = {"test": "attack_run", "category": "attack"}

        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            memory_labels=memory_labels,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            await attack_run.run_async()

            # Check that memory_labels were passed in the call
            call_kwargs = mock_exec.call_args.kwargs
            assert "memory_labels" in call_kwargs
            assert call_kwargs["memory_labels"] == memory_labels

    @pytest.mark.asyncio
    async def test_run_async_passes_attack_execute_params(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, sample_attack_results
    ):
        """Test that attack execute parameters are passed to the executor."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            custom_param="value",
            max_retries=3,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            await attack_run.run_async()

            # Check that custom parameters were passed
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["custom_param"] == "value"
            assert call_kwargs["max_retries"] == 3

    @pytest.mark.asyncio
    async def test_run_async_passes_dataset_params(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, sample_attack_results
    ):
        """Test that dataset parameters are passed to the executor."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            await attack_run.run_async()

            # Check that objectives from dataset were passed
            call_kwargs = mock_exec.call_args.kwargs
            assert "objectives" in call_kwargs
            assert call_kwargs["objectives"] == ["objective1", "objective2", "objective3"]

    @pytest.mark.asyncio
    async def test_run_async_handles_execution_failure(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config
    ):
        """Test that execution failures are properly handled and raised."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception("Execution error")

            with pytest.raises(ValueError, match="Failed to execute attack run"):
                await attack_run.run_async()


@pytest.mark.usefixtures("patch_central_database")
class TestAttackRunIntegration:
    """Integration tests for AttackRun with real configuration files."""

    def test_attack_run_with_real_configs(self, mock_target: PromptTarget):
        """Test creating an attack run with real configuration files from ConfigurationPaths."""
        attack_run = AttackRun(
            attack_config=ConfigurationPaths.attack.foundry.ascii_art,
            dataset_config=ConfigurationPaths.dataset.harm_bench,
            objective_target=mock_target,
        )

        assert attack_run._attack is not None
        assert attack_run._dataset_params is not None
        assert len(attack_run._dataset_params["objectives"]) > 0

    @pytest.mark.asyncio
    async def test_full_attack_run_execution_flow(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config
    ):
        """Test the complete attack run execution flow end-to-end."""
        memory_labels = {"test": "integration", "attack_run": "full"}

        attack_run = AttackRun(
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

            results = await attack_run.run_async(max_concurrency=3)

            # Verify results
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.objective == f"objective{i+1}"
                assert result.outcome == AttackOutcome.SUCCESS

            # Verify the call was made with all expected parameters
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == attack_run._attack
            assert call_kwargs["objectives"] == ["objective1", "objective2", "objective3"]
            assert call_kwargs["memory_labels"] == memory_labels


@pytest.mark.usefixtures("patch_central_database")
class TestAttackRunAdditionalConverters:
    """Tests for AttackRun with additional_request_converters functionality."""

    def test_init_with_single_additional_converter(
        self,
        mock_target: PromptTarget,
        valid_attack_config,
        valid_dataset_config,
        valid_converter_config_base64,
    ):
        """Test initialization with a single additional converter."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            additional_request_converters=[valid_converter_config_base64],
        )

        # Verify the attack was created
        assert attack_run._attack is not None

        # Verify that additional converters were applied
        assert hasattr(attack_run._attack, "_request_converters")
        # Should have the original converter + 1 additional
        assert len(attack_run._attack._request_converters) == 2  # type: ignore

        # Verify the additional converter is of the correct type
        additional_converter = attack_run._attack._request_converters[1].converters[0]  # type: ignore
        assert isinstance(additional_converter, Base64Converter)

    def test_init_with_multiple_additional_converters(
        self,
        mock_target: PromptTarget,
        valid_attack_config,
        valid_dataset_config,
        valid_converter_config_base64,
        valid_converter_config_leetspeak,
    ):
        """Test initialization with multiple additional converters."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            additional_request_converters=[valid_converter_config_base64, valid_converter_config_leetspeak],
        )

        # Verify the attack was created
        assert attack_run._attack is not None

        # Verify that additional converters were applied
        assert hasattr(attack_run._attack, "_request_converters")
        # Should have the original converter + 2 additional
        assert len(attack_run._attack._request_converters) == 3  # type: ignore

        # Verify the additional converters are of the correct types
        converter1 = attack_run._attack._request_converters[1].converters[0]  # type: ignore
        converter2 = attack_run._attack._request_converters[2].converters[0]  # type: ignore
        assert isinstance(converter1, Base64Converter)
        assert isinstance(converter2, LeetspeakConverter)

    def test_init_with_additional_converters_using_configuration_paths(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config
    ):
        """Test initialization with additional converters from ConfigurationPaths."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            additional_request_converters=[
                ConfigurationPaths.converter.base64,
                ConfigurationPaths.converter.leetspeak,
            ],
        )

        # Verify the attack was created
        assert attack_run._attack is not None

        # Verify that additional converters were applied
        assert hasattr(attack_run._attack, "_request_converters")
        # Should have the original converter + 2 additional
        assert len(attack_run._attack._request_converters) == 3  # type: ignore

    def test_init_with_string_paths_for_converters(
        self,
        mock_target: PromptTarget,
        valid_attack_config,
        valid_dataset_config,
        valid_converter_config_base64,
    ):
        """Test initialization with string paths for additional converters."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            additional_request_converters=[str(valid_converter_config_base64)],
        )

        # Verify the attack was created and converters applied
        assert attack_run._attack is not None
        assert len(attack_run._attack._request_converters) == 2  # type: ignore

    def test_init_without_additional_converters(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config
    ):
        """Test initialization without additional converters (default behavior)."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        # Verify the attack was created
        assert attack_run._attack is not None

        # Should only have the original converter from the attack config
        assert len(attack_run._attack._request_converters) == 1  # type: ignore

    def test_init_with_empty_additional_converters_list(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config
    ):
        """Test initialization with an empty list of additional converters."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            additional_request_converters=[],
        )

        # Verify the attack was created
        assert attack_run._attack is not None

        # Should only have the original converter from the attack config
        assert len(attack_run._attack._request_converters) == 1  # type: ignore

    def test_init_with_nonexistent_converter_config(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config
    ):
        """Test initialization with a non-existent converter configuration file."""
        nonexistent_path = Path("/nonexistent/converter.py")

        with pytest.raises(ValueError) as exc_info:
            AttackRun(
                attack_config=valid_attack_config,
                dataset_config=valid_dataset_config,
                objective_target=mock_target,
                additional_request_converters=[nonexistent_path],
            )

        assert "Failed to load additional converter" in str(exc_info.value)
        assert "not found" in str(exc_info.value).lower()

    def test_init_with_invalid_converter_config_no_attribute(
        self,
        mock_target: PromptTarget,
        valid_attack_config,
        valid_dataset_config,
        invalid_converter_config_no_attribute,
    ):
        """Test initialization with converter config missing 'additional_converter' attribute."""
        with pytest.raises(ValueError) as exc_info:
            AttackRun(
                attack_config=valid_attack_config,
                dataset_config=valid_dataset_config,
                objective_target=mock_target,
                additional_request_converters=[invalid_converter_config_no_attribute],
            )

        assert "Failed to load additional converter" in str(exc_info.value)
        assert "additional_converter" in str(exc_info.value)

    def test_init_with_invalid_converter_config_wrong_type(
        self,
        mock_target: PromptTarget,
        valid_attack_config,
        valid_dataset_config,
        invalid_converter_config_wrong_type,
    ):
        """Test initialization with converter config having wrong type for 'additional_converter'."""
        with pytest.raises(ValueError) as exc_info:
            AttackRun(
                attack_config=valid_attack_config,
                dataset_config=valid_dataset_config,
                objective_target=mock_target,
                additional_request_converters=[invalid_converter_config_wrong_type],
            )

        assert "Failed to load additional converter" in str(exc_info.value)
        assert "PromptConverter" in str(exc_info.value)

    def test_load_converter_from_config(
        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config, valid_converter_config_base64
    ):
        """Test the _load_converter_from_config method directly."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
        )

        # Test loading a valid converter
        converter = attack_run._load_converter_from_config(valid_converter_config_base64)
        assert isinstance(converter, Base64Converter)

    def test_additional_converters_with_real_attack_configs(self, mock_target: PromptTarget):
        """Test additional converters with real attack configuration from ConfigurationPaths."""
        attack_run = AttackRun(
            attack_config=ConfigurationPaths.attack.foundry.ascii_art,
            dataset_config=ConfigurationPaths.dataset.harm_bench,
            objective_target=mock_target,
            additional_request_converters=[
                ConfigurationPaths.converter.base64,
                ConfigurationPaths.converter.rot13,
            ],
        )

        assert attack_run._attack is not None
        assert hasattr(attack_run._attack, "_request_converters")
        # Original converters + 2 additional
        initial_count = len(attack_run._attack._request_converters)  # type: ignore
        assert initial_count >= 2  # At least the additional converters should be present

    @pytest.mark.asyncio
    async def test_attack_run_execution_with_additional_converters(
        self,
        mock_target: PromptTarget,
        valid_attack_config,
        valid_dataset_config,
        valid_converter_config_base64,
    ):
        """Test that attack runs successfully with additional converters applied."""
        attack_run = AttackRun(
            attack_config=valid_attack_config,
            dataset_config=valid_dataset_config,
            objective_target=mock_target,
            additional_request_converters=[valid_converter_config_base64],
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

            results = await attack_run.run_async(max_concurrency=1)

            # Verify execution was successful
            assert len(results) == 3
            # Verify the attack has the additional converter
            assert len(attack_run._attack._request_converters) == 2  # type: ignore
