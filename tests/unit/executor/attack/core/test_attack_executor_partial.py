# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for AttackExecutor partial execution result functionality."""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.attack import AttackExecutor, AttackStrategy
from pyrit.executor.attack.core import AttackExecutorResult
from pyrit.models import AttackOutcome, AttackResult


@pytest.fixture
def sample_objectives():
    """Create sample objectives for testing."""
    return ["objective1", "objective2", "objective3", "objective4", "objective5"]


@pytest.fixture
def mock_attack_strategy():
    """Create a mock attack strategy for testing."""
    strategy = MagicMock(spec=AttackStrategy)
    strategy.execute_async = AsyncMock()
    strategy.get_identifier.return_value = {
        "__type__": "TestAttack",
        "__module__": "pyrit.executor.attack.test_attack",
        "id": str(uuid.uuid4()),
    }
    return strategy


def create_attack_result(objective: str, outcome: AttackOutcome = AttackOutcome.SUCCESS) -> AttackResult:
    """Helper to create attack results."""
    return AttackResult(
        conversation_id=str(uuid.uuid4()),
        objective=objective,
        attack_identifier={
            "__type__": "TestAttack",
            "__module__": "test",
            "id": str(uuid.uuid4()),
        },
        outcome=outcome,
        executed_turns=1,
    )


@pytest.mark.asyncio
class TestPartialExecutionWithFailures:
    """Tests for AttackExecutor returning partial results when some objectives fail."""

    async def test_execute_multi_objective_returns_partial_on_some_failures(
        self, mock_attack_strategy, sample_objectives
    ):
        """Test that execute_multi_objective_attack_async returns AttackExecutorResult
        when some objectives fail and return_partial_on_failure=True."""

        # Set up mock to succeed for first 3, fail for last 2
        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)  # Simulate async work
            if objective in ["objective4", "objective5"]:
                raise ValueError(f"Failed to execute {objective}")
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=5)
        result = await executor.execute_multi_objective_attack_async(
            attack=mock_attack_strategy,
            objectives=sample_objectives,
            return_partial_on_failure=True,
        )

        # Should return AttackExecutorResult
        assert isinstance(result, AttackExecutorResult)

        # Should have 3 completed results
        assert len(result.completed_results) == 3
        assert all(r.objective in ["objective1", "objective2", "objective3"] for r in result.completed_results)

        # Should have 2 incomplete objectives
        assert len(result.incomplete_objectives) == 2
        incomplete_objs = [obj for obj, _ in result.incomplete_objectives]
        assert "objective4" in incomplete_objs
        assert "objective5" in incomplete_objs

        # Should have exceptions recorded
        for obj, exc in result.incomplete_objectives:
            assert isinstance(exc, ValueError)
            assert f"Failed to execute {obj}" in str(exc)

        # Helper properties should work
        assert result.has_incomplete is True
        assert result.all_completed is False

    async def test_execute_multi_objective_returns_partial_result_on_all_success(
        self, mock_attack_strategy, sample_objectives
    ):
        """Test that execute_multi_objective_attack_async returns AttackExecutorResult
        even when all objectives succeed, with no incomplete objectives."""

        # Set up mock to succeed for all
        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=5)
        result = await executor.execute_multi_objective_attack_async(
            attack=mock_attack_strategy,
            objectives=sample_objectives,
            return_partial_on_failure=True,
        )

        # Should return AttackExecutorResult even when all succeed
        assert isinstance(result, AttackExecutorResult)
        assert len(result) == 5  # Test that __len__ works
        assert len(result.completed_results) == 5
        assert len(result.incomplete_objectives) == 0
        assert result.all_completed is True
        assert result.has_incomplete is False

        # Test that iteration works
        result_list = list(result)
        assert len(result_list) == 5
        assert all(isinstance(r, AttackResult) for r in result_list)
        assert all(r.objective in sample_objectives for r in result_list)

        # Test that indexing works
        assert isinstance(result[0], AttackResult)
        assert result[0].objective in sample_objectives

    async def test_execute_multi_objective_raises_on_failure_by_default(self, mock_attack_strategy, sample_objectives):
        """Test that execute_multi_objective_attack_async raises exception by default
        (return_partial_on_failure=False by default)."""

        # Set up mock to fail for one objective
        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)
            if objective == "objective3":
                raise ValueError(f"Failed to execute {objective}")
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=5)

        # Default behavior should raise
        with pytest.raises(ValueError, match="Failed to execute objective3"):
            await executor.execute_multi_objective_attack_async(
                attack=mock_attack_strategy,
                objectives=sample_objectives,
            )

    async def test_execute_multi_objective_raises_on_failure_when_explicit_false(
        self, mock_attack_strategy, sample_objectives
    ):
        """Test that execute_multi_objective_attack_async raises exception when
        return_partial_on_failure=False is explicitly set."""

        # Set up mock to fail for one objective
        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)
            if objective == "objective3":
                raise ValueError(f"Failed to execute {objective}")
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=5)

        with pytest.raises(ValueError, match="Failed to execute objective3"):
            await executor.execute_multi_objective_attack_async(
                attack=mock_attack_strategy,
                objectives=sample_objectives,
                return_partial_on_failure=False,
            )

    async def test_execute_single_turn_returns_partial_on_some_failures(self, mock_attack_strategy, sample_objectives):
        """Test that execute_single_turn_attacks_async returns AttackExecutorResult
        when some objectives fail."""

        # Set up mock to succeed for first 2, fail for rest
        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)
            if objective not in ["objective1", "objective2"]:
                raise RuntimeError(f"Single turn failed: {objective}")
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=3)
        result = await executor.execute_single_turn_attacks_async(
            attack=mock_attack_strategy,
            objectives=sample_objectives,
            return_partial_on_failure=True,
        )

        # Should return AttackExecutorResult
        assert isinstance(result, AttackExecutorResult)
        assert len(result.completed_results) == 2
        assert len(result.incomplete_objectives) == 3
        assert result.has_incomplete is True

    async def test_execute_multi_turn_returns_partial_on_some_failures(self, mock_attack_strategy, sample_objectives):
        """Test that execute_multi_turn_attacks_async returns AttackExecutorResult
        when some objectives fail."""

        # Set up mock to fail for middle objectives
        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)
            if objective in ["objective2", "objective3"]:
                raise ConnectionError(f"Multi turn connection failed: {objective}")
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=2)
        result = await executor.execute_multi_turn_attacks_async(
            attack=mock_attack_strategy,
            objectives=sample_objectives,
            return_partial_on_failure=True,
        )

        # Should return AttackExecutorResult
        assert isinstance(result, AttackExecutorResult)
        assert len(result.completed_results) == 3
        assert len(result.incomplete_objectives) == 2

        # Verify the right objectives failed
        incomplete_objs = [obj for obj, _ in result.incomplete_objectives]
        assert "objective2" in incomplete_objs
        assert "objective3" in incomplete_objs

    async def test_partial_result_preserves_exception_types(self, mock_attack_strategy, sample_objectives):
        """Test that AttackExecutorResult preserves the actual exception types."""

        # Set up mock with different exception types
        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)
            if objective == "objective2":
                raise ValueError("Value error in objective2")
            elif objective == "objective4":
                raise RuntimeError("Runtime error in objective4")
            elif objective == "objective5":
                raise ConnectionError("Connection error in objective5")
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=5)
        result = await executor.execute_multi_objective_attack_async(
            attack=mock_attack_strategy,
            objectives=sample_objectives,
            return_partial_on_failure=True,
        )

        # Should have preserved exception types
        assert isinstance(result, AttackExecutorResult)
        assert len(result.incomplete_objectives) == 3

        exception_map = {obj: exc for obj, exc in result.incomplete_objectives}
        assert isinstance(exception_map["objective2"], ValueError)
        assert isinstance(exception_map["objective4"], RuntimeError)
        assert isinstance(exception_map["objective5"], ConnectionError)

    async def test_completed_results_saved_before_failure(self, mock_attack_strategy):
        """Test that completed results are returned even if later objectives fail."""
        objectives = ["fast1", "fast2", "slow_fail", "fast3"]

        completed = []

        async def mock_execute(objective, **kwargs):
            if objective.startswith("fast"):
                await asyncio.sleep(0.01)
                result = create_attack_result(objective)
                completed.append(objective)
                return result
            else:
                await asyncio.sleep(0.02)
                raise Exception("Intentional failure")

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=4)
        result = await executor.execute_multi_objective_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
            return_partial_on_failure=True,
        )

        # All fast objectives should have completed before the slow failure
        assert isinstance(result, AttackExecutorResult)
        assert len(result.completed_results) >= 2  # At least fast1 and fast2
        assert len(result.incomplete_objectives) == 1
        assert result.incomplete_objectives[0][0] == "slow_fail"


@pytest.mark.asyncio
class TestPartialExecutionResultMethods:
    """Tests for AttackExecutorResult convenience methods."""

    async def test_raise_if_incomplete_raises_when_incomplete(self, mock_attack_strategy):
        """Test that raise_if_incomplete() raises the first exception when there are incomplete objectives."""

        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)
            if objective == "objective2":
                raise ValueError("First failure")
            elif objective == "objective3":
                raise RuntimeError("Second failure")
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=5)
        result = await executor.execute_multi_objective_attack_async(
            attack=mock_attack_strategy,
            objectives=["objective1", "objective2", "objective3"],
            return_partial_on_failure=True,
        )

        # Should raise the first exception
        with pytest.raises(ValueError, match="First failure"):
            result.raise_if_incomplete()

    async def test_raise_if_incomplete_does_not_raise_when_complete(self, mock_attack_strategy):
        """Test that raise_if_incomplete() does not raise when all objectives complete."""

        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=5)
        result = await executor.execute_multi_objective_attack_async(
            attack=mock_attack_strategy,
            objectives=["objective1", "objective2"],
            return_partial_on_failure=True,
        )

        # Should not raise
        result.raise_if_incomplete()  # Should complete without error

    async def test_get_results_raises_when_incomplete(self, mock_attack_strategy):
        """Test that get_results() raises when there are incomplete objectives."""

        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)
            if objective == "objective2":
                raise ValueError("Failure in objective2")
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=5)
        result = await executor.execute_multi_objective_attack_async(
            attack=mock_attack_strategy,
            objectives=["objective1", "objective2"],
            return_partial_on_failure=True,
        )

        # Should raise
        with pytest.raises(ValueError, match="Failure in objective2"):
            result.get_results()

    async def test_get_results_returns_list_when_complete(self, mock_attack_strategy):
        """Test that get_results() returns list when all objectives complete."""

        async def mock_execute(objective, **kwargs):
            await asyncio.sleep(0.01)
            return create_attack_result(objective)

        mock_attack_strategy.execute_async = mock_execute

        executor = AttackExecutor(max_concurrency=5)
        result = await executor.execute_multi_objective_attack_async(
            attack=mock_attack_strategy,
            objectives=["objective1", "objective2"],
            return_partial_on_failure=True,
        )

        # Should return list of results
        results_list = result.get_results()
        assert isinstance(results_list, list)
        assert len(results_list) == 2
        assert all(isinstance(r, AttackResult) for r in results_list)
