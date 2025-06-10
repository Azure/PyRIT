# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.attacks.base.attack_context import SingleTurnAttackContext
from pyrit.attacks.base.attack_executor import AttackExecutor
from pyrit.attacks.base.attack_result import AttackOutcome, AttackResult
from pyrit.attacks.base.attack_strategy import AttackStrategy
from pyrit.exceptions.exception_classes import AttackExecutionException


@pytest.fixture
def mock_attack_strategy():
    """Create a mock attack strategy for testing"""
    strategy = MagicMock(spec=AttackStrategy)
    strategy.execute_async = AsyncMock()
    strategy.get_identifier.return_value = {
        "__type__": "TestAttack",
        "__module__": "pyrit.attacks.test_attack",
        "id": str(uuid.uuid4()),
    }
    return strategy


@pytest.fixture
def basic_context():
    """Create a basic context for testing"""
    return SingleTurnAttackContext(
        objective="Test objective", conversation_id=str(uuid.uuid4()), max_attempts_on_failure=2
    )


@pytest.fixture
def sample_attack_result():
    """Create a sample attack result for testing"""
    return AttackResult(
        conversation_id=str(uuid.uuid4()),
        objective="Test objective",
        attack_identifier={
            "__type__": "TestAttack",
            "__module__": "pyrit.attacks.test_attack",
            "id": str(uuid.uuid4()),
        },
        outcome=AttackOutcome.SUCCESS,
        outcome_reason="Objective achieved successfully",
        executed_turns=1,
    )


@pytest.fixture
def multiple_objectives():
    """Create a list of multiple objectives for testing"""
    return [
        "Objective 1: Extract sensitive information",
        "Objective 2: Bypass security controls",
        "Objective 3: Execute unauthorized commands",
    ]


@pytest.mark.usefixtures("patch_central_database")
class TestAttackExecutorInitialization:
    """Tests for AttackExecutor initialization and configuration"""

    def test_init_with_default_max_concurrency(self):
        executor = AttackExecutor()

        assert executor._max_concurrency == 1

    def test_init_with_custom_max_concurrency(self):
        executor = AttackExecutor(max_concurrency=10)

        assert executor._max_concurrency == 10

    @pytest.mark.parametrize(
        "invalid_concurrency,expected_error",
        [
            (0, "max_concurrency must be a positive integer, got 0"),
            (-1, "max_concurrency must be a positive integer, got -1"),
            (-10, "max_concurrency must be a positive integer, got -10"),
        ],
    )
    def test_init_raises_error_for_invalid_concurrency(self, invalid_concurrency, expected_error):
        with pytest.raises(ValueError, match=expected_error):
            AttackExecutor(max_concurrency=invalid_concurrency)

    def test_init_with_maximum_concurrency(self):
        executor = AttackExecutor(max_concurrency=1000)

        assert executor._max_concurrency == 1000


@pytest.mark.usefixtures("patch_central_database")
class TestExecuteMultiObjectiveAttack:
    """Tests for execute_multi_objective_attack method"""

    @pytest.mark.asyncio
    async def test_execute_single_objective(self, mock_attack_strategy, basic_context, sample_attack_result):
        executor = AttackExecutor(max_concurrency=5)

        # Mock duplicate to return a new context
        duplicated_context = MagicMock()
        duplicated_context.objective = None  # Will be set by executor
        basic_context.duplicate = MagicMock(return_value=duplicated_context)

        mock_attack_strategy.execute_async.return_value = sample_attack_result

        results = await executor.execute_multi_objective_attack(
            attack=mock_attack_strategy, context_template=basic_context, objectives=["Single objective"]
        )

        assert len(results) == 1
        assert results[0] == sample_attack_result

        # Verify duplicate was called
        basic_context.duplicate.assert_called_once()

        # Verify objective was set on duplicated context
        assert duplicated_context.objective == "Single objective"

        # Verify execute_async was called with duplicated context
        mock_attack_strategy.execute_async.assert_called_once_with(context=duplicated_context)

    @pytest.mark.asyncio
    async def test_execute_multiple_objectives(self, mock_attack_strategy, basic_context, multiple_objectives):
        executor = AttackExecutor(max_concurrency=5)

        # Create duplicated contexts for each objective
        duplicated_contexts = []
        for i in range(len(multiple_objectives)):
            ctx = MagicMock()
            ctx.objective = None
            duplicated_contexts.append(ctx)

        basic_context.duplicate = MagicMock(side_effect=duplicated_contexts)

        # Create unique results for each objective - using side_effect with a function
        async def create_result(context):
            # Find which objective this is based on the context
            for i, ctx in enumerate(duplicated_contexts):
                if context == ctx:
                    return AttackResult(
                        conversation_id=str(uuid.uuid4()),
                        objective=multiple_objectives[i],
                        attack_identifier={
                            "__type__": "TestAttack",
                            "__module__": "pyrit.attacks.test_attack",
                            "id": str(uuid.uuid4()),
                        },
                        outcome=AttackOutcome.SUCCESS if i % 2 == 0 else AttackOutcome.FAILURE,
                        outcome_reason="Success" if i % 2 == 0 else "Failed",
                        executed_turns=i + 1,
                    )
            raise ValueError("Unknown context")

        mock_attack_strategy.execute_async.side_effect = create_result

        results = await executor.execute_multi_objective_attack(
            attack=mock_attack_strategy, context_template=basic_context, objectives=multiple_objectives
        )

        assert len(results) == len(multiple_objectives)

        # Verify duplicate was called for each objective
        assert basic_context.duplicate.call_count == len(multiple_objectives)

        # Verify each duplicated context had its objective set
        for i, ctx in enumerate(duplicated_contexts):
            assert ctx.objective == multiple_objectives[i]

        # Verify execute_async was called for each context
        assert mock_attack_strategy.execute_async.call_count == len(multiple_objectives)

        # Verify results match expected pattern
        for i, result in enumerate(results):
            assert result.objective == multiple_objectives[i]
            assert result.outcome == (AttackOutcome.SUCCESS if i % 2 == 0 else AttackOutcome.FAILURE)
            assert result.executed_turns == i + 1

    @pytest.mark.asyncio
    async def test_execute_empty_objectives_list(self, mock_attack_strategy, basic_context):
        executor = AttackExecutor(max_concurrency=5)

        # Mock the duplicate method on the context
        original_duplicate = basic_context.duplicate
        basic_context.duplicate = MagicMock(side_effect=original_duplicate)

        results = await executor.execute_multi_objective_attack(
            attack=mock_attack_strategy, context_template=basic_context, objectives=[]
        )

        assert results == []
        mock_attack_strategy.execute_async.assert_not_called()

        # Verify duplicate was never called since there are no objectives
        basic_context.duplicate.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_duplication_isolation(self, mock_attack_strategy, basic_context):
        executor = AttackExecutor(max_concurrency=5)

        # Set up initial context state
        basic_context.memory_labels = {"original": "value"}
        basic_context.metadata = {"test": "data"}

        # Create truly independent duplicated contexts
        duplicated_contexts = []

        def create_duplicate():
            ctx = MagicMock()
            ctx.objective = None
            ctx.memory_labels = {"original": "value"}  # Copy of original
            ctx.metadata = {"test": "data"}  # Copy of original
            duplicated_contexts.append(ctx)
            return ctx

        basic_context.duplicate = MagicMock(side_effect=create_duplicate)

        objectives = ["Objective 1", "Objective 2"]
        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_multi_objective_attack(
            attack=mock_attack_strategy, context_template=basic_context, objectives=objectives
        )

        # Verify duplicate was called for each objective
        assert basic_context.duplicate.call_count == 2

        # Verify each context was independent
        assert len(duplicated_contexts) == 2
        context1, context2 = duplicated_contexts

        # Contexts should be different objects
        assert context1 is not context2
        assert context1 is not basic_context
        assert context2 is not basic_context

        # But should have the same initial values
        assert context1.memory_labels == {"original": "value"}
        assert context2.memory_labels == {"original": "value"}
        assert context1.metadata == {"test": "data"}
        assert context2.metadata == {"test": "data"}

        # And objectives should be set correctly
        assert context1.objective == "Objective 1"
        assert context2.objective == "Objective 2"

    @pytest.mark.asyncio
    async def test_preserves_result_order(self, mock_attack_strategy, basic_context):
        executor = AttackExecutor(max_concurrency=2)

        objectives = [f"Objective {i}" for i in range(5)]

        # Create duplicated contexts
        duplicated_contexts = []
        for _ in objectives:
            ctx = MagicMock()
            ctx.objective = None
            duplicated_contexts.append(ctx)

        basic_context.duplicate = MagicMock(side_effect=duplicated_contexts)

        # Simulate varying execution times
        async def mock_execute(context):
            # Find which context this is
            for i, ctx in enumerate(duplicated_contexts):
                if context == ctx:
                    delay = 0.1 if i in [1, 3] else 0.01
                    await asyncio.sleep(delay)
                    return AttackResult(
                        conversation_id=str(uuid.uuid4()),
                        objective=objectives[i],
                        attack_identifier={
                            "__type__": "TestAttack",
                            "__module__": "pyrit.attacks.test_attack",
                            "id": str(uuid.uuid4()),
                        },
                        outcome=AttackOutcome.SUCCESS,
                        executed_turns=1,
                    )
            raise ValueError("Unknown context")

        mock_attack_strategy.execute_async.side_effect = mock_execute

        results = await executor.execute_multi_objective_attack(
            attack=mock_attack_strategy, context_template=basic_context, objectives=objectives
        )

        # Results should be in the same order as objectives despite varying execution times
        assert len(results) == len(objectives)
        for i, result in enumerate(results):
            assert result.objective == objectives[i]


@pytest.mark.usefixtures("patch_central_database")
class TestConcurrencyControl:
    """Tests for concurrency control in attack execution"""

    @pytest.mark.asyncio
    async def test_respects_max_concurrency_limit(self, mock_attack_strategy, basic_context):
        max_concurrency = 2
        executor = AttackExecutor(max_concurrency=max_concurrency)

        # Mock duplicate method
        basic_context.duplicate = MagicMock(side_effect=lambda: MagicMock(objective=None))

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0

        async def mock_execute(context):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate work
            concurrent_count -= 1
            return MagicMock()

        mock_attack_strategy.execute_async.side_effect = mock_execute

        objectives = [f"Objective {i}" for i in range(10)]

        await executor.execute_multi_objective_attack(
            attack=mock_attack_strategy, context_template=basic_context, objectives=objectives
        )

        assert max_concurrent <= max_concurrency
        assert mock_attack_strategy.execute_async.call_count == len(objectives)

    @pytest.mark.asyncio
    async def test_single_concurrency_serializes_execution(self, mock_attack_strategy, basic_context):
        executor = AttackExecutor(max_concurrency=1)

        # Create duplicated contexts with tracking
        duplicated_contexts = []

        def create_duplicate():
            ctx = MagicMock()
            ctx.objective = None
            duplicated_contexts.append(ctx)
            return ctx

        basic_context.duplicate = MagicMock(side_effect=create_duplicate)

        execution_order = []

        async def mock_execute(context):
            execution_order.append(f"start_{context.objective}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end_{context.objective}")
            return MagicMock()

        mock_attack_strategy.execute_async.side_effect = mock_execute

        objectives = ["A", "B", "C"]

        await executor.execute_multi_objective_attack(
            attack=mock_attack_strategy, context_template=basic_context, objectives=objectives
        )

        # With max_concurrency=1, executions should not overlap
        expected_order = ["start_A", "end_A", "start_B", "end_B", "start_C", "end_C"]
        assert execution_order == expected_order


@pytest.mark.usefixtures("patch_central_database")
class TestErrorHandling:
    """Tests for error handling in attack execution"""

    @pytest.mark.asyncio
    async def test_propagates_strategy_execution_errors(self, mock_attack_strategy, basic_context):
        executor = AttackExecutor(max_concurrency=5)

        basic_context.duplicate = MagicMock(return_value=MagicMock(objective=None))

        mock_attack_strategy.execute_async.side_effect = AttackExecutionException(
            message="Strategy execution failed", attack_name="MockStrategy", objective="Test objective"
        )

        with pytest.raises(AttackExecutionException, match="Strategy execution failed"):
            await executor.execute_multi_objective_attack(
                attack=mock_attack_strategy, context_template=basic_context, objectives=["Test objective"]
            )

    @pytest.mark.asyncio
    async def test_handles_partial_failures(self, mock_attack_strategy, basic_context):
        executor = AttackExecutor(max_concurrency=5)

        # Create contexts for each objective
        contexts = []
        for _ in range(3):
            ctx = MagicMock()
            ctx.objective = None
            contexts.append(ctx)

        basic_context.duplicate = MagicMock(side_effect=contexts)

        # Define execution behavior based on context
        async def mock_execute(context):
            if context == contexts[0]:
                return AttackResult(
                    conversation_id=str(uuid.uuid4()),
                    objective="Success 1",
                    attack_identifier={
                        "__type__": "TestAttack",
                        "__module__": "pyrit.attacks.test_attack",
                        "id": str(uuid.uuid4()),
                    },
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=1,
                )
            elif context == contexts[1]:
                raise RuntimeError("Execution failed")
            else:
                return AttackResult(
                    conversation_id=str(uuid.uuid4()),
                    objective="Success 2",
                    attack_identifier={
                        "__type__": "TestAttack",
                        "__module__": "pyrit.attacks.test_attack",
                        "id": str(uuid.uuid4()),
                    },
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=1,
                )

        mock_attack_strategy.execute_async.side_effect = mock_execute

        objectives = ["Success 1", "Failure", "Success 2"]

        # The failure should propagate
        with pytest.raises(RuntimeError, match="Execution failed"):
            await executor.execute_multi_objective_attack(
                attack=mock_attack_strategy, context_template=basic_context, objectives=objectives
            )


@pytest.mark.usefixtures("patch_central_database")
class TestExecuteParallel:
    """Tests for the internal _execute_parallel method"""

    @pytest.mark.asyncio
    async def test_execute_parallel_with_multiple_contexts(self, mock_attack_strategy):
        executor = AttackExecutor(max_concurrency=5)

        contexts = [
            SingleTurnAttackContext(objective=f"Objective {i}", conversation_id=str(uuid.uuid4())) for i in range(3)
        ]

        # Create results for each context
        async def create_result(context):
            return AttackResult(
                conversation_id=context.conversation_id,
                objective=context.objective,
                attack_identifier={
                    "__type__": "TestAttack",
                    "__module__": "pyrit.attacks.test_attack",
                    "id": str(uuid.uuid4()),
                },
                outcome=AttackOutcome.SUCCESS,
                executed_turns=1,
            )

        mock_attack_strategy.execute_async.side_effect = create_result

        results = await executor._execute_parallel(attack=mock_attack_strategy, contexts=contexts)

        assert len(results) == len(contexts)
        assert mock_attack_strategy.execute_async.call_count == len(contexts)

        # Verify each result matches its context
        for i, result in enumerate(results):
            assert result.conversation_id == contexts[i].conversation_id
            assert result.objective == contexts[i].objective

    @pytest.mark.asyncio
    async def test_execute_parallel_with_empty_contexts(self, mock_attack_strategy):
        executor = AttackExecutor(max_concurrency=5)

        results = await executor._execute_parallel(attack=mock_attack_strategy, contexts=[])

        assert results == []
        mock_attack_strategy.execute_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_parallel_maintains_semaphore_integrity(self, mock_attack_strategy):
        executor = AttackExecutor(max_concurrency=2)

        # Track semaphore state
        active_tasks = []

        async def mock_execute(context):
            task_id = id(asyncio.current_task())
            active_tasks.append(task_id)
            assert len(active_tasks) <= 2  # Should never exceed max_concurrency
            await asyncio.sleep(0.05)
            active_tasks.remove(task_id)
            return MagicMock()

        mock_attack_strategy.execute_async.side_effect = mock_execute

        contexts = [
            SingleTurnAttackContext(objective=f"Objective {i}", conversation_id=str(uuid.uuid4())) for i in range(5)
        ]

        await executor._execute_parallel(attack=mock_attack_strategy, contexts=contexts)

        assert len(active_tasks) == 0  # All tasks should be complete


@pytest.mark.usefixtures("patch_central_database")
class TestIntegrationScenarios:
    """Tests for integration scenarios and edge cases"""

    @pytest.mark.asyncio
    async def test_large_scale_execution(self, mock_attack_strategy, basic_context):
        executor = AttackExecutor(max_concurrency=10)

        # Mock duplicate to create new contexts
        basic_context.duplicate = MagicMock(side_effect=lambda: MagicMock(objective=None))

        # Test with a large number of objectives
        objectives = [f"Objective {i}" for i in range(100)]

        async def mock_execute(context):
            await asyncio.sleep(0.001)  # Small delay to simulate work
            return AttackResult(
                conversation_id=str(uuid.uuid4()),
                objective=context.objective,
                attack_identifier={
                    "__type__": "TestAttack",
                    "__module__": "pyrit.attacks.test_attack",
                    "id": str(uuid.uuid4()),
                },
                outcome=AttackOutcome.SUCCESS,
                executed_turns=1,
            )

        mock_attack_strategy.execute_async.side_effect = mock_execute

        results = await executor.execute_multi_objective_attack(
            attack=mock_attack_strategy, context_template=basic_context, objectives=objectives
        )

        assert len(results) == 100
        assert all(isinstance(r, AttackResult) for r in results)
        assert basic_context.duplicate.call_count == 100

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure_results(self, mock_attack_strategy, basic_context):
        executor = AttackExecutor(max_concurrency=3)

        objectives = [f"Objective {i}" for i in range(6)]

        # Create contexts
        contexts = []
        for _ in objectives:
            ctx = MagicMock()
            ctx.objective = None
            contexts.append(ctx)

        basic_context.duplicate = MagicMock(side_effect=contexts)

        # Alternate between success and failure
        async def mock_execute(context):
            # Find index based on objective
            idx = int(context.objective.split()[-1])
            return AttackResult(
                conversation_id=str(uuid.uuid4()),
                objective=context.objective,
                attack_identifier={
                    "__type__": "TestAttack",
                    "__module__": "pyrit.attacks.test_attack",
                    "id": str(uuid.uuid4()),
                },
                outcome=AttackOutcome.SUCCESS if idx % 2 == 0 else AttackOutcome.FAILURE,
                executed_turns=idx + 1,
            )

        mock_attack_strategy.execute_async.side_effect = mock_execute

        results = await executor.execute_multi_objective_attack(
            attack=mock_attack_strategy, context_template=basic_context, objectives=objectives
        )

        # Verify alternating pattern
        for i, result in enumerate(results):
            assert result.outcome == (AttackOutcome.SUCCESS if i % 2 == 0 else AttackOutcome.FAILURE)
            assert result.executed_turns == i + 1

    @pytest.mark.asyncio
    async def test_context_modifications_during_execution(self, mock_attack_strategy, basic_context):
        executor = AttackExecutor(max_concurrency=2)

        # Track context states during execution
        captured_contexts = []

        # Create contexts with initial state
        contexts = []
        for _ in range(3):
            ctx = MagicMock()
            ctx.objective = None
            ctx.memory_labels = {"original": "value"}
            ctx.achieved_objective = False
            contexts.append(ctx)

        basic_context.duplicate = MagicMock(side_effect=contexts)

        async def mock_execute(context):
            # Capture the context state
            captured_contexts.append(
                {
                    "objective": context.objective,
                    "memory_labels": dict(context.memory_labels),
                    "achieved_objective": context.achieved_objective,
                }
            )
            # Modify the context during execution
            context.achieved_objective = True
            context.memory_labels["modified"] = "yes"
            return MagicMock()

        mock_attack_strategy.execute_async.side_effect = mock_execute

        objectives = ["Obj1", "Obj2", "Obj3"]

        await executor.execute_multi_objective_attack(
            attack=mock_attack_strategy, context_template=basic_context, objectives=objectives
        )

        # Verify each context started with clean state
        assert len(captured_contexts) == 3
        for i, captured in enumerate(captured_contexts):
            assert captured["objective"] == objectives[i]
            assert captured["achieved_objective"] is False
            assert "modified" not in captured["memory_labels"]
