# Copyright (c) Microsoft Corporation.# Copyright (c) Microsoft Corporation.

# Licensed under the MIT license.# Licensed under the MIT license.



"""Tests for the scenarios.Scenario class.""""""Tests for the scenarios.AttackRun class."""



from unittest.mock import AsyncMock, MagicMock, patchimport tempfile

from pathlib import Path

import pytestfrom unittest.mock import AsyncMock, MagicMock, patch



from pyrit.models import AttackOutcome, AttackResultimport pytest

from pyrit.prompt_target import PromptTarget

from pyrit.scenarios import AttackRun, Scenariofrom pyrit.executor.attack import AttackExecutor

from pyrit.models import AttackOutcome, AttackResult

from pyrit.prompt_target import PromptTarget

@pytest.fixturefrom pyrit.scenarios import AttackRun

def mock_target():from pyrit.setup import ConfigurationPaths

    """Create a mock PromptTarget for testing."""

    return MagicMock(spec=PromptTarget)

@pytest.fixture

def mock_target():

@pytest.fixture    """Create a mock PromptTarget for testing."""

def mock_attack_runs():    return MagicMock(spec=PromptTarget)

    """Create mock AttackRun instances for testing."""

    run1 = MagicMock(spec=AttackRun)

    run2 = MagicMock(spec=AttackRun)@pytest.fixture

    run3 = MagicMock(spec=AttackRun)def valid_attack_config():

    return [run1, run2, run3]    """Create a temporary valid attack configuration file."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:

        f.write(

@pytest.fixture            """

def sample_attack_results():from pyrit.executor.attack import AttackConverterConfig

    """Create sample attack results for testing."""from pyrit.prompt_converter import AsciiArtConverter

    return [from pyrit.prompt_normalizer import PromptConverterConfiguration

        AttackResult(

            conversation_id=f"conv-{i}",converters = PromptConverterConfiguration.from_converters(converters=[AsciiArtConverter()])

            objective=f"objective{i}",attack_converter_config = AttackConverterConfig(request_converters=converters)

            attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": str(i)},

            outcome=AttackOutcome.SUCCESS,attack_config = {

            executed_turns=1,    "attack_type": "PromptSendingAttack",

        )    "attack_converter_config": attack_converter_config,

        for i in range(5)}

    ]"""

        )

        temp_path = f.name

@pytest.mark.usefixtures("patch_central_database")

class TestScenarioInitialization:    yield Path(temp_path)

    """Tests for Scenario class initialization."""    Path(temp_path).unlink()



    def test_init_with_valid_params(self, mock_attack_runs):

        """Test successful initialization with valid parameters."""@pytest.fixture

        scenario = Scenario(def valid_dataset_config():

            name="Test Scenario",    """Create a temporary valid dataset configuration file."""

            attack_runs=mock_attack_runs,    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:

        )        f.write(

            """

        assert scenario.name == "Test Scenario"dataset_config = {

        assert scenario.attack_run_count == 3    "objectives": ["objective1", "objective2", "objective3"],

        assert scenario._attack_runs == mock_attack_runs}

        assert scenario._memory_labels == {}"""

        )

    def test_init_with_memory_labels(self, mock_attack_runs):        temp_path = f.name

        """Test initialization with memory labels."""

        memory_labels = {"test": "scenario", "category": "foundry"}    yield Path(temp_path)

    Path(temp_path).unlink()

        scenario = Scenario(

            name="Test Scenario",

            attack_runs=mock_attack_runs,@pytest.fixture

            memory_labels=memory_labels,def sample_attack_results():

        )    """Create sample attack results for testing."""

    return [

        assert scenario._memory_labels == memory_labels        AttackResult(

            conversation_id="conv-1",

    def test_init_fails_with_empty_attack_runs(self):            objective="objective1",

        """Test that initialization fails when attack_runs list is empty."""            attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "1"},

        with pytest.raises(ValueError, match="Scenario must contain at least one AttackRun"):            outcome=AttackOutcome.SUCCESS,

            Scenario(            executed_turns=1,

                name="Empty Scenario",        ),

                attack_runs=[],        AttackResult(

            )            conversation_id="conv-2",

            objective="objective2",

    def test_name_property(self, mock_attack_runs):            attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "2"},

        """Test the name property getter."""            outcome=AttackOutcome.SUCCESS,

        scenario = Scenario(            executed_turns=1,

            name="My Scenario",        ),

            attack_runs=mock_attack_runs,        AttackResult(

        )            conversation_id="conv-3",

            objective="objective3",

        assert scenario.name == "My Scenario"            attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "3"},

            outcome=AttackOutcome.FAILURE,

    def test_attack_run_count_property(self, mock_attack_runs):            executed_turns=1,

        """Test the attack_run_count property."""        ),

        scenario = Scenario(    ]

            name="Test Scenario",

            attack_runs=mock_attack_runs,

        )@pytest.mark.usefixtures("patch_central_database")

class TestAttackRunInitialization:

        assert scenario.attack_run_count == len(mock_attack_runs)    """Tests for AttackRun class initialization."""



    def test_init_with_valid_configs(self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config):

@pytest.mark.usefixtures("patch_central_database")        """Test successful initialization with valid configuration files."""

class TestScenarioExecution:        attack_run = AttackRun(

    """Tests for Scenario execution methods."""            attack_config=valid_attack_config,

            dataset_config=valid_dataset_config,

    @pytest.mark.asyncio            objective_target=mock_target,

    async def test_run_async_executes_all_runs(self, mock_attack_runs, sample_attack_results):        )

        """Test that run_async executes all attack runs sequentially."""

        # Configure mock attack runs to return results        assert attack_run._objective_target == mock_target

        for i, run in enumerate(mock_attack_runs):        assert attack_run._attack_config_path == valid_attack_config

            run.run_async = AsyncMock(return_value=[sample_attack_results[i]])        assert attack_run._dataset_config_path == valid_dataset_config

        assert attack_run._memory_labels == {}

        scenario = Scenario(        assert attack_run._attack is not None

            name="Test Scenario",        assert attack_run._dataset_params is not None

            attack_runs=mock_attack_runs,

        )    def test_init_with_memory_labels(self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config):

        """Test initialization with memory labels."""

        results = await scenario.run_async()        memory_labels = {"test": "label", "category": "attack"}



        # Verify all runs were executed        attack_run = AttackRun(

        for run in mock_attack_runs:            attack_config=valid_attack_config,

            run.run_async.assert_called_once_with(max_concurrency=1)            dataset_config=valid_dataset_config,

            objective_target=mock_target,

        # Verify results are aggregated            memory_labels=memory_labels,

        assert len(results) == 3        )

        assert results[0] == sample_attack_results[0]

        assert results[1] == sample_attack_results[1]        assert attack_run._memory_labels == memory_labels

        assert results[2] == sample_attack_results[2]

    def test_init_with_attack_execute_params(

    @pytest.mark.asyncio        self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config

    async def test_run_async_with_custom_concurrency(self, mock_attack_runs, sample_attack_results):    ):

        """Test that max_concurrency is passed to each attack run."""        """Test initialization with additional attack execute parameters."""

        for i, run in enumerate(mock_attack_runs):        attack_run = AttackRun(

            run.run_async = AsyncMock(return_value=[sample_attack_results[i]])            attack_config=valid_attack_config,

            dataset_config=valid_dataset_config,

        scenario = Scenario(            objective_target=mock_target,

            name="Test Scenario",            max_retries=5,

            attack_runs=mock_attack_runs,            custom_param="value",

        )        )



        await scenario.run_async(max_concurrency=5)        assert attack_run._attack_execute_params["max_retries"] == 5

        assert attack_run._attack_execute_params["custom_param"] == "value"

        # Verify max_concurrency was passed to each run

        for run in mock_attack_runs:    def test_init_with_string_paths(self, mock_target: PromptTarget, valid_attack_config, valid_dataset_config):

            run.run_async.assert_called_once_with(max_concurrency=5)        """Test initialization with string paths instead of Path objects."""

        attack_run = AttackRun(

    @pytest.mark.asyncio            attack_config=str(valid_attack_config),

    async def test_run_async_aggregates_multiple_results(self, mock_attack_runs, sample_attack_results):            dataset_config=str(valid_dataset_config),

        """Test that results from multiple attack runs are properly aggregated."""            objective_target=mock_target,

        # Configure runs to return different numbers of results        )

        mock_attack_runs[0].run_async = AsyncMock(return_value=sample_attack_results[0:2])

        mock_attack_runs[1].run_async = AsyncMock(return_value=sample_attack_results[2:4])        assert attack_run._attack_config_path == valid_attack_config

        mock_attack_runs[2].run_async = AsyncMock(return_value=sample_attack_results[4:5])        assert attack_run._dataset_config_path == valid_dataset_config



        scenario = Scenario(    def test_init_with_configuration_paths(self, mock_target: PromptTarget):

            name="Test Scenario",        """Test initialization with ConfigurationPaths."""

            attack_runs=mock_attack_runs,        attack_run = AttackRun(

        )            attack_config=ConfigurationPaths.attack.foundry.ascii_art,

            dataset_config=ConfigurationPaths.dataset.harm_bench,

        results = await scenario.run_async()            objective_target=mock_target,

        )

        # Verify all results are aggregated correctly

        assert len(results) == 5        assert attack_run._attack is not None

        for i, result in enumerate(results):        assert attack_run._dataset_params is not None

            assert result == sample_attack_results[i]        assert "objectives" in attack_run._dataset_params



    @pytest.mark.asyncio

    async def test_run_async_stops_on_failure(self, mock_attack_runs):@pytest.mark.usefixtures("patch_central_database")

        """Test that execution stops when an attack run fails."""class TestAttackRunValidation:

        # Configure first run to succeed, second to fail    """Tests for AttackRun validation methods."""

        mock_attack_runs[0].run_async = AsyncMock(return_value=[])

        mock_attack_runs[1].run_async = AsyncMock(side_effect=Exception("Attack failed"))    def test_init_fails_with_nonexistent_attack_config(self, mock_target: PromptTarget, valid_dataset_config):

        mock_attack_runs[2].run_async = AsyncMock(return_value=[])        """Test that initialization fails when attack config file doesn't exist."""

        with pytest.raises(FileNotFoundError, match="Attack configuration file not found"):

        scenario = Scenario(            AttackRun(

            name="Test Scenario",                attack_config="nonexistent_attack.py",

            attack_runs=mock_attack_runs,                dataset_config=valid_dataset_config,

        )                objective_target=mock_target,

            )

        with pytest.raises(ValueError, match="Failed to execute attack run 2"):

            await scenario.run_async()    def test_init_fails_with_nonexistent_dataset_config(self, mock_target: PromptTarget, valid_attack_config):

        """Test that initialization fails when dataset config file doesn't exist."""

        # Verify first run was executed but third was not        with pytest.raises(FileNotFoundError, match="Dataset configuration file not found"):

        mock_attack_runs[0].run_async.assert_called_once()            AttackRun(

        mock_attack_runs[1].run_async.assert_called_once()                attack_config=valid_attack_config,

        mock_attack_runs[2].run_async.assert_not_called()                dataset_config="nonexistent_dataset.py",

                objective_target=mock_target,

    @pytest.mark.asyncio            )

    async def test_run_async_with_single_attack_run(self, sample_attack_results):

        """Test execution with a single attack run."""    def test_init_fails_with_directory_as_attack_config(

        single_run = MagicMock(spec=AttackRun)        self, mock_target: PromptTarget, valid_dataset_config, tmp_path

        single_run.run_async = AsyncMock(return_value=sample_attack_results[0:2])    ):

        """Test that initialization fails when attack config path is a directory."""

        scenario = Scenario(        with pytest.raises(ValueError, match="Attack configuration path is not a file"):

            name="Single Run Scenario",            AttackRun(

            attack_runs=[single_run],                attack_config=tmp_path,

        )                dataset_config=valid_dataset_config,

                objective_target=mock_target,

        results = await scenario.run_async()            )



        assert len(results) == 2    def test_init_fails_with_directory_as_dataset_config(

        single_run.run_async.assert_called_once()        self, mock_target: PromptTarget, valid_attack_config, tmp_path

    ):

        """Test that initialization fails when dataset config path is a directory."""

@pytest.mark.usefixtures("patch_central_database")        with pytest.raises(ValueError, match="Dataset configuration path is not a file"):

class TestScenarioIntegration:            AttackRun(

    """Integration tests for Scenario with real AttackRun instances."""                attack_config=valid_attack_config,

                dataset_config=tmp_path,

    @pytest.mark.asyncio                objective_target=mock_target,

    async def test_scenario_execution_order(self, sample_attack_results):            )

        """Test that attack runs are executed in the correct order."""

        execution_order = []    def test_init_fails_with_invalid_attack_config(self, mock_target: PromptTarget, valid_dataset_config):

        """Test that initialization fails with invalid attack configuration."""

        async def create_tracked_run(run_id: int):        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:

            """Create a run that tracks when it executes."""            f.write("# Invalid config - no attack_config dictionary\n")

            run = MagicMock(spec=AttackRun)            temp_path = f.name



            async def track_execution(*args, **kwargs):        try:

                execution_order.append(run_id)            with pytest.raises(ValueError, match="Failed to create attack from configuration file"):

                return [sample_attack_results[run_id]]                AttackRun(

                    attack_config=temp_path,

            run.run_async = track_execution                    dataset_config=valid_dataset_config,

            return run                    objective_target=mock_target,

                )

        runs = [await create_tracked_run(i) for i in range(3)]        finally:

            Path(temp_path).unlink()

        scenario = Scenario(

            name="Order Test",    def test_init_fails_with_invalid_dataset_config(self, mock_target: PromptTarget, valid_attack_config):

            attack_runs=runs,        """Test that initialization fails with invalid dataset configuration."""

        )        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:

            f.write("# Invalid config - no dataset_config dictionary\n")

        await scenario.run_async()            temp_path = f.name



        # Verify runs executed in order        try:

        assert execution_order == [0, 1, 2]            with pytest.raises(ValueError, match="Failed to load dataset from configuration file"):

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
