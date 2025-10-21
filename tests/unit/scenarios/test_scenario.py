# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the scenarios.Scenario class."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.models import AttackOutcome, AttackResult
from pyrit.scenarios import AttackRun, Scenario
from pyrit.scenarios.scenario import ScenarioIdentifier, ScenarioResult


@pytest.fixture
def mock_attack_runs():
    """Create mock AttackRun instances for testing."""
    run1 = MagicMock(spec=AttackRun)
    run2 = MagicMock(spec=AttackRun)
    run3 = MagicMock(spec=AttackRun)
    return [run1, run2, run3]


@pytest.fixture
def sample_attack_results():
    """Create sample attack results for testing."""
    return [
        AttackResult(
            conversation_id=f"conv-{i}",
            objective=f"objective{i}",
            attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": str(i)},
            outcome=AttackOutcome.SUCCESS,
            executed_turns=1,
        )
        for i in range(5)
    ]


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioInitialization:
    """Tests for Scenario class initialization."""

    def test_init_with_valid_params(self, mock_attack_runs):
        """Test successful initialization with valid parameters."""
        scenario = Scenario(
            name="Test Scenario",
            version=1,
            attack_runs=mock_attack_runs,
        )

        assert scenario.name == "Test Scenario"
        assert scenario.attack_run_count == 3
        assert scenario._attack_runs == mock_attack_runs
        assert scenario._memory_labels == {}
        assert scenario._identifier.name == "Scenario"
        assert scenario._identifier.version == 1

    def test_init_with_memory_labels(self, mock_attack_runs):
        """Test initialization with memory labels."""
        memory_labels = {"test": "scenario", "category": "foundry"}

        scenario = Scenario(
            name="Test Scenario",
            version=2,
            attack_runs=mock_attack_runs,
            memory_labels=memory_labels,
        )

        assert scenario._memory_labels == memory_labels

    def test_init_creates_scenario_identifier(self, mock_attack_runs):
        """Test that initialization creates a proper ScenarioIdentifier."""
        scenario = Scenario(
            name="Test Scenario",
            version=3,
            attack_runs=mock_attack_runs,
        )

        assert isinstance(scenario._identifier, ScenarioIdentifier)
        assert scenario._identifier.name == "Scenario"
        assert scenario._identifier.version == 3
        assert scenario._identifier.pyrit_version is not None

    def test_init_fails_with_empty_attack_runs(self):
        """Test that initialization fails when attack_runs list is empty."""
        with pytest.raises(ValueError, match="Scenario must contain at least one AttackRun"):
            Scenario(
                name="Test Scenario",
                version=1,
                attack_runs=[],
            )


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioExecution:
    """Tests for Scenario execution methods."""

    @pytest.mark.asyncio
    async def test_run_async_executes_all_runs(self, mock_attack_runs, sample_attack_results):
        """Test that run_async executes all attack runs sequentially."""
        # Configure each run to return different results
        for i, run in enumerate(mock_attack_runs):
            run.run_async = AsyncMock(return_value=[sample_attack_results[i]])

        scenario = Scenario(
            name="Test Scenario",
            version=1,
            attack_runs=mock_attack_runs,
        )

        result = await scenario.run_async()

        # Verify return type is ScenarioResult
        assert isinstance(result, ScenarioResult)

        # Verify all runs were executed
        assert len(result.attack_results) == 3
        for run in mock_attack_runs:
            run.run_async.assert_called_once()

        # Verify results are aggregated correctly
        assert result.attack_results[0] == sample_attack_results[0]
        assert result.attack_results[1] == sample_attack_results[1]
        assert result.attack_results[2] == sample_attack_results[2]

    @pytest.mark.asyncio
    async def test_run_async_with_custom_concurrency(self, mock_attack_runs, sample_attack_results):
        """Test that max_concurrency is passed to each attack run."""
        for i, run in enumerate(mock_attack_runs):
            run.run_async = AsyncMock(return_value=[sample_attack_results[i]])

        scenario = Scenario(
            name="Test Scenario",
            version=1,
            attack_runs=mock_attack_runs,
        )

        result = await scenario.run_async(max_concurrency=5)

        # Verify max_concurrency was passed to each run
        for run in mock_attack_runs:
            run.run_async.assert_called_once_with(max_concurrency=5)

        # Verify result structure
        assert isinstance(result, ScenarioResult)
        assert len(result.attack_results) == 3

    @pytest.mark.asyncio
    async def test_run_async_aggregates_multiple_results(self, mock_attack_runs, sample_attack_results):
        """Test that results from multiple attack runs are properly aggregated."""
        # Configure runs to return different numbers of results
        mock_attack_runs[0].run_async = AsyncMock(return_value=sample_attack_results[0:2])
        mock_attack_runs[1].run_async = AsyncMock(return_value=sample_attack_results[2:4])
        mock_attack_runs[2].run_async = AsyncMock(return_value=sample_attack_results[4:5])

        scenario = Scenario(
            name="Test Scenario",
            version=1,
            attack_runs=mock_attack_runs,
        )

        result = await scenario.run_async()

        # Should have 5 total results (2 + 2 + 1)
        assert isinstance(result, ScenarioResult)
        assert len(result.attack_results) == 5
        assert result.attack_results == sample_attack_results

    @pytest.mark.asyncio
    async def test_run_async_stops_on_error(self, mock_attack_runs, sample_attack_results):
        """Test that execution stops when an attack run fails."""
        mock_attack_runs[0].run_async = AsyncMock(return_value=[sample_attack_results[0]])
        mock_attack_runs[1].run_async = AsyncMock(side_effect=Exception("Test error"))
        mock_attack_runs[2].run_async = AsyncMock(return_value=[sample_attack_results[2]])

        scenario = Scenario(
            name="Test Scenario",
            version=1,
            attack_runs=mock_attack_runs,
        )

        with pytest.raises(ValueError, match="Failed to execute attack run 2 in scenario 'Test Scenario'"):
            await scenario.run_async()

        # First run should have been executed
        mock_attack_runs[0].run_async.assert_called_once()
        # Second run should have been attempted
        mock_attack_runs[1].run_async.assert_called_once()
        # Third run should not have been executed
        mock_attack_runs[2].run_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_with_default_concurrency(self, mock_attack_runs, sample_attack_results):
        """Test that default concurrency (1) is used when not specified."""
        for i, run in enumerate(mock_attack_runs):
            run.run_async = AsyncMock(return_value=[sample_attack_results[i]])

        scenario = Scenario(
            name="Test Scenario",
            version=1,
            attack_runs=mock_attack_runs,
        )

        result = await scenario.run_async()

        # Verify default concurrency of 1 was passed
        for run in mock_attack_runs:
            run.run_async.assert_called_once_with(max_concurrency=1)

        # Verify result structure
        assert isinstance(result, ScenarioResult)

    @pytest.mark.asyncio
    async def test_run_async_returns_scenario_result_with_identifier(self, mock_attack_runs, sample_attack_results):
        """Test that run_async returns ScenarioResult with proper identifier."""
        for i, run in enumerate(mock_attack_runs):
            run.run_async = AsyncMock(return_value=[sample_attack_results[i]])

        scenario = Scenario(
            name="Test Scenario",
            version=5,
            attack_runs=mock_attack_runs,
        )

        result = await scenario.run_async()

        assert isinstance(result, ScenarioResult)
        assert isinstance(result.scenario_identifier, ScenarioIdentifier)
        assert result.scenario_identifier.name == "Scenario"
        assert result.scenario_identifier.version == 5
        assert result.scenario_identifier.pyrit_version is not None


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioProperties:
    """Tests for Scenario property methods."""

    def test_name_property(self, mock_attack_runs):
        """Test that name property returns the scenario name."""
        scenario = Scenario(
            name="My Test Scenario",
            version=1,
            attack_runs=mock_attack_runs,
        )

        assert scenario.name == "My Test Scenario"

    def test_attack_run_count_property(self, mock_attack_runs):
        """Test that attack_run_count returns the correct count."""
        scenario = Scenario(
            name="Test Scenario",
            version=1,
            attack_runs=mock_attack_runs,
        )

        assert scenario.attack_run_count == 3

    def test_attack_run_count_with_different_sizes(self):
        """Test attack_run_count with different numbers of runs."""
        single_run = [MagicMock(spec=AttackRun)]
        scenario1 = Scenario(name="Single", version=1, attack_runs=single_run)  # type: ignore
        assert scenario1.attack_run_count == 1

        many_runs = [MagicMock(spec=AttackRun) for _ in range(10)]
        scenario2 = Scenario(name="Many", version=1, attack_runs=many_runs)  # type: ignore
        assert scenario2.attack_run_count == 10


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioResult:
    """Tests for ScenarioResult class."""

    def test_scenario_result_initialization(self, sample_attack_results):
        """Test ScenarioResult initialization."""
        identifier = ScenarioIdentifier(name="TestScenario", version=1)
        result = ScenarioResult(scenario_identifier=identifier, attack_results=sample_attack_results)

        assert result.scenario_identifier == identifier
        assert result.attack_results == sample_attack_results
        assert len(result.attack_results) == 5

    def test_scenario_result_with_empty_results(self):
        """Test ScenarioResult with empty attack results."""
        identifier = ScenarioIdentifier(name="TestScenario", version=1)
        result = ScenarioResult(scenario_identifier=identifier, attack_results=[])

        assert len(result.attack_results) == 0


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioIdentifier:
    """Tests for ScenarioIdentifier class."""

    def test_scenario_identifier_initialization(self):
        """Test ScenarioIdentifier initialization."""
        identifier = ScenarioIdentifier(name="TestScenario", version=2)

        assert identifier.name == "TestScenario"
        assert identifier.version == 2
        assert identifier.pyrit_version is not None

    def test_scenario_identifier_with_custom_pyrit_version(self):
        """Test ScenarioIdentifier with custom pyrit version."""
        identifier = ScenarioIdentifier(name="TestScenario", version=1, pyrit_version="1.2.3")

        assert identifier.pyrit_version == "1.2.3"

    def test_scenario_identifier_with_init_data(self):
        """Test ScenarioIdentifier with init_data."""
        init_data = {"param1": "value1", "param2": 42}
        identifier = ScenarioIdentifier(name="TestScenario", version=1, init_data=init_data)

        assert identifier.init_data == init_data
