# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the scenarios.Scenario class."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.models import AttackOutcome, AttackResult
from pyrit.scenarios import AttackRun, Scenario
from pyrit.scenarios.attack_run import AttackRunResult
from pyrit.scenarios.scenario import ScenarioIdentifier, ScenarioResult


@pytest.fixture
def mock_attack_runs():
    """Create mock AttackRun instances for testing."""
    run1 = MagicMock(spec=AttackRun)
    run1.attack_run_name = "attack_run_1"
    run2 = MagicMock(spec=AttackRun)
    run2.attack_run_name = "attack_run_2"
    run3 = MagicMock(spec=AttackRun)
    run3.attack_run_name = "attack_run_3"
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


class ConcreteScenario(Scenario):
    """Concrete implementation of Scenario for testing."""

    def __init__(self, attack_runs_to_return=None, **kwargs):
        super().__init__(**kwargs)
        self._attack_runs_to_return = attack_runs_to_return or []

    async def _get_attack_runs_async(self):
        return self._attack_runs_to_return


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioInitialization:
    """Tests for Scenario class initialization."""

    def test_init_with_valid_params(self):
        """Test successful initialization with valid parameters."""
        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
        )

        assert scenario.name == "Test Scenario"
        assert scenario._identifier.name == "ConcreteScenario"
        assert scenario._identifier.version == 1
        assert scenario._memory_labels == {}
        assert scenario._max_concurrency == 1
        assert scenario.attack_run_count == 0  # Not initialized yet

    def test_init_with_memory_labels(self):
        """Test initialization with memory labels."""
        memory_labels = {"test": "scenario", "category": "foundry"}

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=2,
            memory_labels=memory_labels,
        )

        assert scenario._memory_labels == memory_labels

    def test_init_with_custom_concurrency(self):
        """Test initialization with custom max_concurrency."""
        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            max_concurrency=5,
        )

        assert scenario._max_concurrency == 5

    def test_init_creates_scenario_identifier(self):
        """Test that initialization creates a proper ScenarioIdentifier."""
        scenario = ConcreteScenario(
            name="Test Scenario",
            version=3,
        )

        assert isinstance(scenario._identifier, ScenarioIdentifier)
        assert scenario._identifier.name == "ConcreteScenario"
        assert scenario._identifier.version == 3
        assert scenario._identifier.pyrit_version is not None

    def test_init_with_empty_attack_strategies(self):
        """Test that initialization works without attack_strategies."""
        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
        )

        # Test that scenario initializes correctly without attack_strategies
        assert scenario.attack_run_count == 0


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioInitialization2:
    """Tests for Scenario initialization_async method."""

    @pytest.mark.asyncio
    async def test_initialize_async_populates_attack_runs(self, mock_attack_runs):
        """Test that initialize_async populates attack runs."""
        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            attack_strategies=["base64"],
            attack_runs_to_return=mock_attack_runs,
        )

        assert scenario.attack_run_count == 0

        await scenario.initialize_async()

        assert scenario.attack_run_count == len(mock_attack_runs)
        assert scenario._attack_runs == mock_attack_runs


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioExecution:
    """Tests for Scenario execution methods."""

    @pytest.mark.asyncio
    async def test_run_async_executes_all_runs(self, mock_attack_runs, sample_attack_results):
        """Test that run_async executes all attack runs sequentially."""
        # Configure each run to return different results
        for i, run in enumerate(mock_attack_runs):
            run.run_async = AsyncMock(
                return_value=AttackRunResult(results=[sample_attack_results[i]], name=run.attack_run_name)
            )

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            attack_runs_to_return=mock_attack_runs,
        )
        await scenario.initialize_async()

        result = await scenario.run_async()

        # Verify return type is ScenarioResult
        assert isinstance(result, ScenarioResult)

        # Verify all runs were executed with correct concurrency
        assert len(result.attack_results) == 3
        for run in mock_attack_runs:
            run.run_async.assert_called_once_with(max_concurrency=1)

        # Verify results are aggregated correctly by attack run name
        assert "attack_run_1" in result.attack_results
        assert "attack_run_2" in result.attack_results
        assert "attack_run_3" in result.attack_results
        assert result.attack_results["attack_run_1"][0] == sample_attack_results[0]
        assert result.attack_results["attack_run_2"][0] == sample_attack_results[1]
        assert result.attack_results["attack_run_3"][0] == sample_attack_results[2]

    @pytest.mark.asyncio
    async def test_run_async_with_custom_concurrency(self, mock_attack_runs, sample_attack_results):
        """Test that max_concurrency from init is passed to each attack run."""
        for i, run in enumerate(mock_attack_runs):
            run.run_async = AsyncMock(
                return_value=AttackRunResult(results=[sample_attack_results[i]], name=run.attack_run_name)
            )

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            max_concurrency=5,
            attack_runs_to_return=mock_attack_runs,
        )
        await scenario.initialize_async()

        result = await scenario.run_async()

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
        mock_attack_runs[0].run_async = AsyncMock(
            return_value=AttackRunResult(results=sample_attack_results[0:2], name=mock_attack_runs[0].attack_run_name)
        )
        mock_attack_runs[1].run_async = AsyncMock(
            return_value=AttackRunResult(results=sample_attack_results[2:4], name=mock_attack_runs[1].attack_run_name)
        )
        mock_attack_runs[2].run_async = AsyncMock(
            return_value=AttackRunResult(results=sample_attack_results[4:5], name=mock_attack_runs[2].attack_run_name)
        )

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            attack_runs_to_return=mock_attack_runs,
        )
        await scenario.initialize_async()

        result = await scenario.run_async()

        # Should have 3 attack runs with results (2 + 2 + 1)
        assert isinstance(result, ScenarioResult)
        assert len(result.attack_results) == 3
        assert len(result.attack_results["attack_run_1"]) == 2
        assert len(result.attack_results["attack_run_2"]) == 2
        assert len(result.attack_results["attack_run_3"]) == 1

    @pytest.mark.asyncio
    async def test_run_async_stops_on_error(self, mock_attack_runs, sample_attack_results):
        """Test that execution stops when an attack run fails."""
        mock_attack_runs[0].run_async = AsyncMock(
            return_value=AttackRunResult(results=[sample_attack_results[0]], name=mock_attack_runs[0].attack_run_name)
        )
        mock_attack_runs[1].run_async = AsyncMock(side_effect=Exception("Test error"))
        mock_attack_runs[2].run_async = AsyncMock(
            return_value=AttackRunResult(results=[sample_attack_results[2]], name=mock_attack_runs[2].attack_run_name)
        )

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            attack_runs_to_return=mock_attack_runs,
        )
        await scenario.initialize_async()

        with pytest.raises(ValueError, match="Failed to execute attack run 2 in scenario 'Test Scenario'"):
            await scenario.run_async()

        # First run should have been executed
        mock_attack_runs[0].run_async.assert_called_once()
        # Second run should have been attempted
        mock_attack_runs[1].run_async.assert_called_once()
        # Third run should not have been executed
        mock_attack_runs[2].run_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_async_fails_without_initialization(self):
        """Test that run_async fails if initialize_async was not called."""
        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
        )

        with pytest.raises(ValueError, match="Cannot run scenario with no attack runs"):
            await scenario.run_async()

    @pytest.mark.asyncio
    async def test_run_async_returns_scenario_result_with_identifier(self, mock_attack_runs, sample_attack_results):
        """Test that run_async returns ScenarioResult with proper identifier."""
        for i, run in enumerate(mock_attack_runs):
            run.run_async = AsyncMock(
                return_value=AttackRunResult(results=[sample_attack_results[i]], name=run.attack_run_name)
            )

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=5,
            attack_runs_to_return=mock_attack_runs,
        )
        await scenario.initialize_async()

        result = await scenario.run_async()

        assert isinstance(result, ScenarioResult)
        assert isinstance(result.scenario_identifier, ScenarioIdentifier)
        assert result.scenario_identifier.name == "ConcreteScenario"
        assert result.scenario_identifier.version == 5
        assert result.scenario_identifier.pyrit_version is not None
        assert result.get_strategies_used() == ["attack_run_1", "attack_run_2", "attack_run_3"]


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioProperties:
    """Tests for Scenario property methods."""

    def test_name_property(self):
        """Test that name property returns the scenario name."""
        scenario = ConcreteScenario(
            name="My Test Scenario",
            version=1,
        )

        assert scenario.name == "My Test Scenario"

    @pytest.mark.asyncio
    async def test_attack_run_count_property(self, mock_attack_runs):
        """Test that attack_run_count returns the correct count."""
        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            attack_runs_to_return=mock_attack_runs,
        )

        assert scenario.attack_run_count == 0

        await scenario.initialize_async()

        assert scenario.attack_run_count == 3

    @pytest.mark.asyncio
    async def test_attack_run_count_with_different_sizes(self):
        """Test attack_run_count with different numbers of runs."""
        single_run = [MagicMock(spec=AttackRun)]
        scenario1 = ConcreteScenario(
            name="Single",
            version=1,
            attack_runs_to_return=single_run,
        )
        await scenario1.initialize_async()
        assert scenario1.attack_run_count == 1

        many_runs = [MagicMock(spec=AttackRun) for _ in range(10)]
        scenario2 = ConcreteScenario(
            name="Many",
            version=1,
            attack_runs_to_return=many_runs,
        )
        await scenario2.initialize_async()
        assert scenario2.attack_run_count == 10


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioResult:
    """Tests for ScenarioResult class."""

    def test_scenario_result_initialization(self, sample_attack_results):
        """Test ScenarioResult initialization."""
        identifier = ScenarioIdentifier(name="Test", scenario_version=1)
        result = ScenarioResult(
            scenario_identifier=identifier,
            objective_target_identifier={"__type__": "TestTarget", "__module__": "test"},
            attack_results={"base64": sample_attack_results[:3], "rot13": sample_attack_results[3:]},
        )

        assert result.scenario_identifier == identifier
        assert result.get_strategies_used() == ["base64", "rot13"]
        assert len(result.attack_results) == 2
        assert len(result.attack_results["base64"]) == 3
        assert len(result.attack_results["rot13"]) == 2

    def test_scenario_result_with_empty_results(self):
        """Test ScenarioResult with empty attack results."""
        identifier = ScenarioIdentifier(name="TestScenario", scenario_version=1)
        result = ScenarioResult(
            scenario_identifier=identifier,
            objective_target_identifier={"__type__": "TestTarget", "__module__": "test"},
            attack_results={"base64": []},
        )

        assert len(result.attack_results["base64"]) == 0
        assert result.objective_achieved_rate() == 0

    def test_scenario_result_objective_achieved_rate(self, sample_attack_results):
        """Test objective_achieved_rate calculation."""
        identifier = ScenarioIdentifier(name="Test", scenario_version=1)

        # All successful
        result = ScenarioResult(
            scenario_identifier=identifier,
            objective_target_identifier={"__type__": "TestTarget", "__module__": "test"},
            attack_results={"base64": sample_attack_results},
        )
        assert result.objective_achieved_rate() == 100

        # Mixed outcomes
        mixed_results = sample_attack_results[:3] + [
            AttackResult(
                conversation_id="conv-fail",
                objective="objective",
                attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "1"},
                outcome=AttackOutcome.FAILURE,
                executed_turns=1,
            ),
            AttackResult(
                conversation_id="conv-fail2",
                objective="objective",
                attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "2"},
                outcome=AttackOutcome.FAILURE,
                executed_turns=1,
            ),
        ]
        result2 = ScenarioResult(
            scenario_identifier=identifier,
            objective_target_identifier={"__type__": "TestTarget", "__module__": "test"},
            attack_results={"base64": mixed_results},
        )
        assert result2.objective_achieved_rate() == 60  # 3 out of 5


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioIdentifier:
    """Tests for ScenarioIdentifier class."""

    def test_scenario_identifier_initialization(self):
        """Test ScenarioIdentifier initialization."""
        identifier = ScenarioIdentifier(name="TestScenario", scenario_version=2)

        assert identifier.name == "TestScenario"
        assert identifier.version == 2
        assert identifier.pyrit_version is not None

    def test_scenario_identifier_with_custom_pyrit_version(self):
        """Test ScenarioIdentifier initialization sets pyrit version automatically."""
        identifier = ScenarioIdentifier(name="TestScenario", scenario_version=1)

        assert identifier.pyrit_version is not None
        assert identifier.name == "TestScenario"

    def test_scenario_identifier_with_init_data(self):
        """Test ScenarioIdentifier with init_data."""
        init_data = {"param1": "value1", "param2": 42}
        identifier = ScenarioIdentifier(name="TestScenario", scenario_version=1, init_data=init_data)

        assert identifier.init_data == init_data
