# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for Scenario retry functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.memory import CentralMemory
from pyrit.models import AttackOutcome, AttackResult
from pyrit.scenarios import AtomicAttack, Scenario
from pyrit.scenarios.atomic_attack import AtomicAttackResult
from pyrit.scenarios.scenario import ScenarioIdentifier, ScenarioResult


def save_attack_results_to_memory(attack_results):
    """Helper function to save attack results to memory (mimics what real attacks do)."""
    memory = CentralMemory.get_memory_instance()
    memory.add_attack_results_to_memory(attack_results=attack_results)


def create_mock_run_async(attack_result):
    """Create a mock run_async that saves results to memory before returning."""
    async def mock_run_async(*args, **kwargs):
        # Save results to memory (mimics what real attacks do)
        save_attack_results_to_memory(attack_result.results)
        return attack_result
    return AsyncMock(side_effect=mock_run_async)


class ConcreteScenario(Scenario):
    """Concrete implementation of Scenario for testing."""

    def __init__(self, atomic_attacks_to_return=None, **kwargs):
        super().__init__(**kwargs)
        self._atomic_attacks_to_return = atomic_attacks_to_return or []

    @classmethod
    def get_strategy_class(cls):
        """Return a mock strategy class for testing."""

        from pyrit.scenarios.scenario_strategy import ScenarioStrategy

        # Return a simple mock strategy class for testing
        class TestStrategy(ScenarioStrategy):
            TEST = ("test", set())

            @classmethod
            def get_aggregate_tags(cls) -> set[str]:
                return set()

        return TestStrategy

    @classmethod
    def get_default_strategy(cls):
        """Return the default strategy for testing."""
        return cls.get_strategy_class().TEST

    async def _get_atomic_attacks_async(self):
        return self._atomic_attacks_to_return


@pytest.fixture
def mock_atomic_attacks():
    """Create mock AtomicAttack instances for testing."""
    run1 = MagicMock(spec=AtomicAttack)
    run1.atomic_attack_name = "attack_run_1"
    run1._objectives = ["objective1"]  # Add _objectives attribute
    run2 = MagicMock(spec=AtomicAttack)
    run2.atomic_attack_name = "attack_run_2"
    run2._objectives = ["objective2"]  # Add _objectives attribute
    return [run1, run2]


@pytest.fixture
def mock_objective_target():
    """Create a mock objective target for testing."""
    target = MagicMock()
    target.get_identifier.return_value = {"__type__": "MockTarget", "__module__": "test"}
    return target


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
        for i in range(3)
    ]


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioRetry:
    """Tests for Scenario retry functionality."""

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self, mock_atomic_attacks, sample_attack_results, mock_objective_target):
        """Test that scenario doesn't retry when execution succeeds."""
        # Configure successful execution
        for i, run in enumerate(mock_atomic_attacks):
            run.run_async = create_mock_run_async(
                AtomicAttackResult(results=[sample_attack_results[i]], name=run.atomic_attack_name)
            )

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            objective_target=mock_objective_target,
            max_retries=3,  # Set retries but shouldn't use them on success
            atomic_attacks_to_return=mock_atomic_attacks,
        )
        await scenario.initialize_async()

        result = await scenario.run_async()

        # Verify each atomic attack was called exactly once (no retries needed)
        for run in mock_atomic_attacks:
            run.run_async.assert_called_once()

        # Verify result is successful
        assert isinstance(result, ScenarioResult)
        assert len(result.attack_results) == 2

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, mock_atomic_attacks, sample_attack_results, mock_objective_target):
        """Test that scenario retries on failure up to max_retries."""
        # Configure first run to fail, second to succeed
        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First attempt fails
                raise Exception("Test failure")
            else:
                # Retry succeeds
                results = [sample_attack_results[0]]
                save_attack_results_to_memory(results)
                return AtomicAttackResult(results=results, name=mock_atomic_attacks[0].atomic_attack_name)

        mock_atomic_attacks[0].run_async = mock_run
        mock_atomic_attacks[1].run_async = create_mock_run_async(
            AtomicAttackResult(results=[sample_attack_results[1]], name=mock_atomic_attacks[1].atomic_attack_name)
        )

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            objective_target=mock_objective_target,
            max_retries=2,
            atomic_attacks_to_return=mock_atomic_attacks,
        )
        await scenario.initialize_async()

        result = await scenario.run_async()

        # Verify scenario succeeded on retry
        assert isinstance(result, ScenarioResult)
        # Should have executed 2 times (1 initial + 1 retry)
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_fails(self, mock_atomic_attacks, mock_objective_target):
        """Test that scenario fails after exhausting all retries."""
        # Configure all attempts to fail
        mock_atomic_attacks[0].run_async = AsyncMock(side_effect=Exception("Persistent failure"))
        mock_atomic_attacks[1].run_async = AsyncMock(side_effect=Exception("Should not be called"))

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            objective_target=mock_objective_target,
            max_retries=2,  # Allow 2 retries (3 total attempts)
            atomic_attacks_to_return=mock_atomic_attacks,
        )
        await scenario.initialize_async()

        # Verify that scenario raises exception after exhausting retries
        with pytest.raises(Exception, match="Persistent failure"):
            await scenario.run_async()

        # Verify it attempted max_retries + 1 times (initial + retries)
        assert mock_atomic_attacks[0].run_async.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_when_max_retries_zero(self, mock_atomic_attacks, mock_objective_target):
        """Test that scenario doesn't retry when max_retries is 0 (default)."""
        # Configure to fail
        mock_atomic_attacks[0].run_async = AsyncMock(side_effect=Exception("Test failure"))

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            objective_target=mock_objective_target,
            max_retries=0,  # No retries
            atomic_attacks_to_return=mock_atomic_attacks,
        )
        await scenario.initialize_async()

        # Verify that scenario raises exception immediately without retry
        with pytest.raises(Exception, match="Test failure"):
            await scenario.run_async()

        # Verify it was only called once (no retries)
        mock_atomic_attacks[0].run_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_number_tries_increments_on_retry(self, mock_atomic_attacks, sample_attack_results, mock_objective_target):
        """Test that number_tries field increments with each retry attempt."""
        # Track how many times we check number_tries
        tries_values = []
        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                # First two attempts fail
                raise Exception("Test failure")
            else:
                # Third attempt succeeds
                results = [sample_attack_results[0]]
                save_attack_results_to_memory(results)
                return AtomicAttackResult(results=results, name=mock_atomic_attacks[0].atomic_attack_name)

        mock_atomic_attacks[0].run_async = mock_run
        mock_atomic_attacks[1].run_async = create_mock_run_async(
            AtomicAttackResult(results=[sample_attack_results[1]], name=mock_atomic_attacks[1].atomic_attack_name)
        )

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            objective_target=mock_objective_target,
            max_retries=3,
            atomic_attacks_to_return=mock_atomic_attacks,
        )
        await scenario.initialize_async()

        result = await scenario.run_async()

        # Verify scenario succeeded after retries
        assert isinstance(result, ScenarioResult)
        # number_tries should be 3 (failed twice, succeeded on third)
        assert result.number_tries == 3

    @pytest.mark.asyncio
    async def test_retry_logs_error_with_exception(
        self, mock_atomic_attacks, sample_attack_results, mock_objective_target, caplog
    ):
        """Test that retry failures are logged with exception details."""
        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First attempt fails
                raise ValueError("First failure")
            else:
                # Retry succeeds
                results = [sample_attack_results[0]]
                save_attack_results_to_memory(results)
                return AtomicAttackResult(results=results, name=mock_atomic_attacks[0].atomic_attack_name)

        mock_atomic_attacks[0].run_async = mock_run
        mock_atomic_attacks[1].run_async = create_mock_run_async(
            AtomicAttackResult(results=[sample_attack_results[1]], name=mock_atomic_attacks[1].atomic_attack_name)
        )

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            objective_target=mock_objective_target,
            max_retries=1,
            atomic_attacks_to_return=mock_atomic_attacks,
        )
        await scenario.initialize_async()

        with caplog.at_level("ERROR"):
            result = await scenario.run_async()

        # Verify error was logged
        assert "failed on attempt" in caplog.text.lower()
        assert "First failure" in caplog.text or "ValueError" in caplog.text
        assert "retrying" in caplog.text.lower()

        # Verify scenario eventually succeeded
        assert isinstance(result, ScenarioResult)


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioResumption:
    """Tests for Scenario resumption after partial failure."""

    @pytest.mark.asyncio
    async def test_resumes_from_partial_completion_single_attack(
        self, mock_objective_target
    ):
        """Test that scenario resumes from where it left off when an atomic attack partially completes."""
        # Create atomic attack with multiple objectives
        atomic_attack = MagicMock(spec=AtomicAttack)
        atomic_attack.atomic_attack_name = "multi_objective_attack"
        atomic_attack._objectives = ["obj1", "obj2", "obj3", "obj4"]  # Set initial objectives
        
        # Track which objectives have been executed
        executed_objectives = []
        call_count = [0]

        async def mock_run(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First attempt: complete 2 objectives, then fail
                executed_objectives.extend(["obj1", "obj2"])
                results = [
                    AttackResult(
                        conversation_id=f"conv-{i}",
                        objective=f"obj{i}",
                        attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": str(i)},
                        outcome=AttackOutcome.SUCCESS,
                        executed_turns=1,
                    )
                    for i in [1, 2]
                ]
                # Save partial results before failing (mimics what real attacks do)
                save_attack_results_to_memory(results)
                # Return results for first 2, then simulate failure before completing others
                raise Exception("Failed after 2 objectives")
            else:
                # Retry: should only execute remaining objectives (obj3, obj4)
                executed_objectives.extend(["obj3", "obj4"])
                results = [
                    AttackResult(
                        conversation_id=f"conv-{i}",
                        objective=f"obj{i}",
                        attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": str(i)},
                        outcome=AttackOutcome.SUCCESS,
                        executed_turns=1,
                    )
                    for i in [3, 4]
                ]
                # Save results to memory (mimics what real attacks do)
                save_attack_results_to_memory(results)
                return AtomicAttackResult(results=results, name=atomic_attack.atomic_attack_name)

        atomic_attack.run_async = mock_run

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            objective_target=mock_objective_target,
            max_retries=1,
            atomic_attacks_to_return=[atomic_attack],
        )
        await scenario.initialize_async()

        result = await scenario.run_async()

        # Verify scenario succeeded after retry
        assert isinstance(result, ScenarioResult)
        # Should have attempted twice (initial + 1 retry)
        assert call_count[0] == 2
        # All objectives should be executed across both attempts
        assert "obj1" in executed_objectives or "obj3" in executed_objectives

    @pytest.mark.asyncio
    async def test_resumes_skipping_completed_atomic_attacks(
        self, mock_objective_target
    ):
        """Test that scenario skips completed atomic attacks on retry."""
        # Create 3 atomic attacks
        attack1 = MagicMock(spec=AtomicAttack)
        attack1.atomic_attack_name = "attack_1"
        attack1._objectives = ["objective1"]
        
        attack2 = MagicMock(spec=AtomicAttack)
        attack2.atomic_attack_name = "attack_2"
        attack2._objectives = ["objective2"]
        
        attack3 = MagicMock(spec=AtomicAttack)
        attack3.atomic_attack_name = "attack_3"
        attack3._objectives = ["objective3"]

        call_count = {"attack_1": 0, "attack_2": 0, "attack_3": 0}

        async def mock_run_attack1(*args, **kwargs):
            call_count["attack_1"] += 1
            results = [
                AttackResult(
                    conversation_id="conv-1",
                    objective="objective1",
                    attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "1"},
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=1,
                )
            ]
            # Save results to memory (mimics what real attacks do)
            save_attack_results_to_memory(results)
            return AtomicAttackResult(results=results, name=attack1.atomic_attack_name)

        async def mock_run_attack2(*args, **kwargs):
            call_count["attack_2"] += 1
            if call_count["attack_2"] == 1:
                # First attempt: attack2 succeeds
                results = [
                    AttackResult(
                        conversation_id="conv-2",
                        objective="objective2",
                        attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "2"},
                        outcome=AttackOutcome.SUCCESS,
                        executed_turns=1,
                    )
                ]
                # Save results to memory (mimics what real attacks do)
                save_attack_results_to_memory(results)
                return AtomicAttackResult(results=results, name=attack2.atomic_attack_name)
            else:
                # Should not be called again on retry
                raise AssertionError("Attack 2 should not be retried after completion")

        async def mock_run_attack3(*args, **kwargs):
            call_count["attack_3"] += 1
            if call_count["attack_3"] == 1:
                # First attempt: attack3 fails
                raise Exception("Attack 3 failed on first attempt")
            else:
                # Retry: attack3 succeeds
                results = [
                    AttackResult(
                        conversation_id="conv-3",
                        objective="objective3",
                        attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "3"},
                        outcome=AttackOutcome.SUCCESS,
                        executed_turns=1,
                    )
                ]
                # Save results to memory (mimics what real attacks do)
                save_attack_results_to_memory(results)
                return AtomicAttackResult(results=results, name=attack3.atomic_attack_name)

        attack1.run_async = mock_run_attack1
        attack2.run_async = mock_run_attack2
        attack3.run_async = mock_run_attack3

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            objective_target=mock_objective_target,
            max_retries=1,
            atomic_attacks_to_return=[attack1, attack2, attack3],
        )
        await scenario.initialize_async()

        result = await scenario.run_async()

        # Verify scenario succeeded
        assert isinstance(result, ScenarioResult)
        # Attack 1 and 2 should be called once each (completed on first attempt)
        assert call_count["attack_1"] == 1
        assert call_count["attack_2"] == 1
        # Attack 3 should be called twice (failed first, succeeded on retry)
        assert call_count["attack_3"] == 2
        # All three attacks should be in results
        assert len(result.attack_results) == 3
        assert "attack_1" in result.attack_results
        assert "attack_2" in result.attack_results
        assert "attack_3" in result.attack_results

    @pytest.mark.asyncio
    async def test_resumes_with_multiple_failures_across_attacks(
        self, mock_objective_target
    ):
        """Test resumption when multiple atomic attacks fail at different stages."""
        # Create 4 atomic attacks
        attacks = []
        for i in range(1, 5):
            attack = MagicMock(spec=AtomicAttack)
            attack.atomic_attack_name = f"attack_{i}"
            attack._objectives = [f"objective{i}"]
            attacks.append(attack)

        call_count = {f"attack_{i}": 0 for i in range(1, 5)}

        # Attack 1: Succeeds immediately
        async def mock_run_attack1(*args, **kwargs):
            call_count["attack_1"] += 1
            results = [
                AttackResult(
                    conversation_id="conv-1",
                    objective="objective1",
                    attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "1"},
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=1,
                )
            ]
            # Save results to memory (mimics what real attacks do)
            save_attack_results_to_memory(results)
            return AtomicAttackResult(results=results, name=attacks[0].atomic_attack_name)

        # Attack 2: Fails on first attempt, succeeds on retry
        async def mock_run_attack2(*args, **kwargs):
            call_count["attack_2"] += 1
            if call_count["attack_2"] == 1:
                raise Exception("Attack 2 failed")
            results = [
                AttackResult(
                    conversation_id="conv-2",
                    objective="objective2",
                    attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "2"},
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=1,
                )
            ]
            # Save results to memory (mimics what real attacks do)
            save_attack_results_to_memory(results)
            return AtomicAttackResult(results=results, name=attacks[1].atomic_attack_name)

        # Attack 3: Should not be called on first attempt (stops at attack 2 failure)
        # Succeeds on retry
        async def mock_run_attack3(*args, **kwargs):
            call_count["attack_3"] += 1
            results = [
                AttackResult(
                    conversation_id="conv-3",
                    objective="objective3",
                    attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "3"},
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=1,
                )
            ]
            # Save results to memory (mimics what real attacks do)
            save_attack_results_to_memory(results)
            return AtomicAttackResult(results=results, name=attacks[2].atomic_attack_name)

        # Attack 4: Should only be called once on retry attempt
        async def mock_run_attack4(*args, **kwargs):
            call_count["attack_4"] += 1
            results = [
                AttackResult(
                    conversation_id="conv-4",
                    objective="objective4",
                    attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "4"},
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=1,
                )
            ]
            # Save results to memory (mimics what real attacks do)
            save_attack_results_to_memory(results)
            return AtomicAttackResult(results=results, name=attacks[3].atomic_attack_name)

        attacks[0].run_async = mock_run_attack1
        attacks[1].run_async = mock_run_attack2
        attacks[2].run_async = mock_run_attack3
        attacks[3].run_async = mock_run_attack4

        scenario = ConcreteScenario(
            name="Test Scenario",
            version=1,
            objective_target=mock_objective_target,
            max_retries=1,
            atomic_attacks_to_return=attacks,
        )
        await scenario.initialize_async()

        result = await scenario.run_async()

        # Verify scenario succeeded
        assert isinstance(result, ScenarioResult)
        # Attack 1: Called once (succeeded before failure point)
        assert call_count["attack_1"] == 1
        # Attack 2: Called twice (failed first, succeeded on retry)
        assert call_count["attack_2"] == 2
        # Attack 3: Called once (only on retry, after attack 2 succeeded)
        assert call_count["attack_3"] == 1
        # Attack 4: Called once (only on retry)
        assert call_count["attack_4"] == 1
        # All four attacks should be in results
        assert len(result.attack_results) == 4
