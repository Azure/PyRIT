# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.attack import (
    AttackExecutor,
    AttackStrategy,
    MultiTurnAttackContext,
    SingleTurnAttackContext,
)
from pyrit.models import AttackOutcome, AttackResult, Message

# Common accepted parameters for single-turn attacks
SINGLE_TURN_ACCEPTED_PARAMS = {
    "objective",
    "memory_labels",
    "prepended_conversation",
    "start_time",
    "related_conversations",
    "conversation_id",
    "next_message",
    "system_prompt",
    "metadata",
}

# Common accepted parameters for multi-turn attacks (includes additional fields)
MULTI_TURN_ACCEPTED_PARAMS = {
    "objective",
    "memory_labels",
    "prepended_conversation",
    "start_time",
    "related_conversations",
    "session",
    "executed_turns",
    "last_response",
    "last_score",
    "next_message",
    "max_turns",  # Common attack param
    "temperature",  # Common attack param
    "messages",  # For multi-prompt attacks
}


@pytest.fixture
def mock_attack_strategy():
    """Create a mock attack strategy for testing"""
    strategy = MagicMock(spec=AttackStrategy)
    strategy.execute_async = AsyncMock()
    strategy.execute_with_context_async = AsyncMock()
    strategy.accepted_context_parameters = SINGLE_TURN_ACCEPTED_PARAMS | MULTI_TURN_ACCEPTED_PARAMS
    strategy.get_identifier.return_value = {
        "__type__": "TestAttack",
        "__module__": "pyrit.executor.attack.test_attack",
        "id": str(uuid.uuid4()),
    }
    return strategy


@pytest.fixture
def basic_context():
    """Create a basic context for testing"""
    return SingleTurnAttackContext(objective="Test objective", conversation_id=str(uuid.uuid4()))


@pytest.fixture
def multi_turn_context():
    """Create a basic multi-turn context for testing"""
    return MultiTurnAttackContext(objective="Test multi-turn objective")


@pytest.fixture
def mock_single_turn_attack_strategy():
    """Create a mock single-turn attack strategy for testing"""
    strategy = MagicMock(spec=AttackStrategy)
    strategy.execute_async = AsyncMock()
    strategy.execute_with_context_async = AsyncMock()
    strategy._context_type = SingleTurnAttackContext
    strategy.accepted_context_parameters = SINGLE_TURN_ACCEPTED_PARAMS
    strategy.get_identifier.return_value = {
        "__type__": "TestSingleTurnAttack",
        "__module__": "pyrit.executor.attack.test_attack",
        "id": str(uuid.uuid4()),
    }
    return strategy


@pytest.fixture
def mock_multi_turn_attack_strategy():
    """Create a mock multi-turn attack strategy for testing"""
    strategy = MagicMock(spec=AttackStrategy)
    strategy.execute_async = AsyncMock()
    strategy.execute_with_context_async = AsyncMock()
    strategy._context_type = MultiTurnAttackContext
    strategy.accepted_context_parameters = MULTI_TURN_ACCEPTED_PARAMS
    strategy.get_identifier.return_value = {
        "__type__": "TestMultiTurnAttack",
        "__module__": "pyrit.executor.attack.test_attack",
        "id": str(uuid.uuid4()),
    }
    return strategy


@pytest.fixture
def sample_messages():
    """Create sample messages for testing"""
    return [
        Message.from_prompt(prompt="First prompt", role="user"),
        Message.from_prompt(prompt="Second prompt", role="user"),
        Message.from_prompt(prompt="Third prompt", role="user"),
    ]


@pytest.fixture
def sample_attack_result():
    """Create a sample attack result for testing"""
    return AttackResult(
        conversation_id=str(uuid.uuid4()),
        objective="Test objective",
        attack_identifier={
            "__type__": "TestAttack",
            "__module__": "pyrit.executor.attack.test_attack",
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
class TestExecuteAttackAsync:
    """Tests for execute_attack_async method - the primary execution method"""

    @pytest.mark.asyncio
    async def test_execute_single_objective(self, mock_attack_strategy, sample_attack_result):
        """Test executing with a single objective"""
        executor = AttackExecutor(max_concurrency=5)
        mock_attack_strategy.execute_async.return_value = sample_attack_result

        results = await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=["Single objective"],
        )

        assert len(results) == 1
        assert results[0] == sample_attack_result
        mock_attack_strategy.execute_async.assert_called_once()

        # Verify correct parameters passed
        call = mock_attack_strategy.execute_async.call_args
        assert call.kwargs["objective"] == "Single objective"

    @pytest.mark.asyncio
    async def test_execute_multiple_objectives(self, mock_attack_strategy, multiple_objectives):
        """Test executing with multiple objectives"""
        executor = AttackExecutor(max_concurrency=5)

        results = []
        for obj in multiple_objectives:
            results.append(
                AttackResult(
                    conversation_id=str(uuid.uuid4()),
                    objective=obj,
                    attack_identifier={"__type__": "TestAttack"},
                    outcome=AttackOutcome.SUCCESS,
                    executed_turns=1,
                )
            )
        mock_attack_strategy.execute_async.side_effect = results

        actual_results = await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=multiple_objectives,
        )

        assert len(actual_results) == len(multiple_objectives)
        assert mock_attack_strategy.execute_async.call_count == len(multiple_objectives)

    @pytest.mark.asyncio
    async def test_execute_with_prepended_conversations(self, mock_attack_strategy, sample_messages):
        """Test that prepended_conversations are passed correctly per-objective"""
        executor = AttackExecutor(max_concurrency=2)
        objectives = ["Obj1", "Obj2"]
        prepended_convs = [[sample_messages[0]], [sample_messages[1]]]

        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
            prepended_conversations=prepended_convs,
        )

        calls = mock_attack_strategy.execute_async.call_args_list
        assert calls[0].kwargs["prepended_conversation"] == prepended_convs[0]
        assert calls[1].kwargs["prepended_conversation"] == prepended_convs[1]

    @pytest.mark.asyncio
    async def test_execute_with_next_messages(self, mock_attack_strategy, sample_messages):
        """Test that next_messages are passed correctly per-objective"""
        executor = AttackExecutor(max_concurrency=2)
        objectives = ["Obj1", "Obj2"]
        next_msgs = [sample_messages[0], sample_messages[1]]

        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
            next_messages=next_msgs,
        )

        calls = mock_attack_strategy.execute_async.call_args_list
        assert calls[0].kwargs["next_message"] == next_msgs[0]
        assert calls[1].kwargs["next_message"] == next_msgs[1]

    @pytest.mark.asyncio
    async def test_execute_with_memory_labels_dict(self, mock_attack_strategy):
        """Test memory_labels as dict broadcasts to all objectives"""
        executor = AttackExecutor(max_concurrency=2)
        objectives = ["Obj1", "Obj2", "Obj3"]
        memory_labels = {"test": "label", "category": "attack"}

        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
            memory_labels=memory_labels,
        )

        # All calls should have the same memory labels
        for call in mock_attack_strategy.execute_async.call_args_list:
            assert call.kwargs["memory_labels"] == memory_labels

    @pytest.mark.asyncio
    async def test_execute_with_memory_labels_list(self, mock_attack_strategy):
        """Test memory_labels as list provides per-objective labels"""
        executor = AttackExecutor(max_concurrency=2)
        objectives = ["Obj1", "Obj2"]
        memory_labels_list = [{"label": "first"}, {"label": "second"}]

        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
            memory_labels=memory_labels_list,
        )

        calls = mock_attack_strategy.execute_async.call_args_list
        assert calls[0].kwargs["memory_labels"] == memory_labels_list[0]
        assert calls[1].kwargs["memory_labels"] == memory_labels_list[1]

    @pytest.mark.asyncio
    async def test_execute_with_per_attack_params(self, mock_attack_strategy):
        """Test per_attack_params provides per-objective parameters"""
        executor = AttackExecutor(max_concurrency=2)
        objectives = ["Obj1", "Obj2"]
        per_attack_params = [{"max_turns": 5}, {"max_turns": 10}]

        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
            per_attack_params=per_attack_params,
        )

        calls = mock_attack_strategy.execute_async.call_args_list
        assert calls[0].kwargs["max_turns"] == 5
        assert calls[1].kwargs["max_turns"] == 10

    @pytest.mark.asyncio
    async def test_execute_with_broadcast_attack_params(self, mock_attack_strategy):
        """Test broadcast_attack_params applied to all objectives"""
        executor = AttackExecutor(max_concurrency=2)
        objectives = ["Obj1", "Obj2"]

        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
            max_turns=5,
            temperature=0.7,
        )

        for call in mock_attack_strategy.execute_async.call_args_list:
            assert call.kwargs["max_turns"] == 5
            assert call.kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_per_attack_params_override_broadcast(self, mock_attack_strategy):
        """Test that per_attack_params override broadcast params"""
        executor = AttackExecutor(max_concurrency=2)
        objectives = ["Obj1", "Obj2"]
        per_attack_params = [{"max_turns": 10}, {}]  # First overrides, second uses broadcast

        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
            per_attack_params=per_attack_params,
            max_turns=5,  # Broadcast param
        )

        calls = mock_attack_strategy.execute_async.call_args_list
        assert calls[0].kwargs["max_turns"] == 10  # Overridden
        assert calls[1].kwargs["max_turns"] == 5  # Uses broadcast

    @pytest.mark.asyncio
    async def test_validates_empty_objectives(self, mock_attack_strategy):
        """Test that empty objectives list raises ValueError"""
        executor = AttackExecutor()

        with pytest.raises(ValueError, match="At least one objective must be provided"):
            await executor.execute_attack_async(
                attack=mock_attack_strategy,
                objectives=[],
            )

    @pytest.mark.asyncio
    async def test_validates_prepended_conversations_length(self, mock_attack_strategy):
        """Test validation of prepended_conversations length"""
        executor = AttackExecutor()
        objectives = ["Obj1", "Obj2"]
        prepended_convs = [[]]  # Wrong length

        with pytest.raises(ValueError, match="Number of prepended_conversations .* must match"):
            await executor.execute_attack_async(
                attack=mock_attack_strategy,
                objectives=objectives,
                prepended_conversations=prepended_convs,
            )

    @pytest.mark.asyncio
    async def test_validates_next_messages_length(self, mock_attack_strategy):
        """Test validation of next_messages length"""
        executor = AttackExecutor()
        objectives = ["Obj1", "Obj2"]
        next_msgs = [Message.from_prompt(prompt="msg", role="user")]  # Wrong length

        with pytest.raises(ValueError, match="Number of next_messages .* must match"):
            await executor.execute_attack_async(
                attack=mock_attack_strategy,
                objectives=objectives,
                next_messages=next_msgs,
            )

    @pytest.mark.asyncio
    async def test_validates_per_attack_params_length(self, mock_attack_strategy):
        """Test validation of per_attack_params length"""
        executor = AttackExecutor()
        objectives = ["Obj1", "Obj2"]
        per_attack_params = [{}]  # Wrong length

        with pytest.raises(ValueError, match="Number of per_attack_params .* must match"):
            await executor.execute_attack_async(
                attack=mock_attack_strategy,
                objectives=objectives,
                per_attack_params=per_attack_params,
            )

    @pytest.mark.asyncio
    async def test_validates_memory_labels_list_length(self, mock_attack_strategy):
        """Test validation of memory_labels list length"""
        executor = AttackExecutor()
        objectives = ["Obj1", "Obj2"]
        memory_labels = [{}]  # Wrong length

        with pytest.raises(ValueError, match="Number of memory_labels .* must match"):
            await executor.execute_attack_async(
                attack=mock_attack_strategy,
                objectives=objectives,
                memory_labels=memory_labels,
            )

    @pytest.mark.asyncio
    async def test_concurrency_control(self, mock_attack_strategy):
        """Test that concurrency is properly limited"""
        max_concurrency = 2
        executor = AttackExecutor(max_concurrency=max_concurrency)

        concurrent_count = 0
        max_concurrent = 0

        async def mock_execute(**kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)
            concurrent_count -= 1
            return MagicMock()

        mock_attack_strategy.execute_async.side_effect = mock_execute

        objectives = [f"Objective {i}" for i in range(10)]

        await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
        )

        assert max_concurrent <= max_concurrency

    @pytest.mark.asyncio
    async def test_partial_failure_with_return_partial(self, mock_attack_strategy):
        """Test return_partial_on_failure=True returns partial results"""
        executor = AttackExecutor(max_concurrency=3)

        async def mock_execute(**kwargs):
            if kwargs["objective"] == "Obj2":
                raise RuntimeError("Execution failed")
            return AttackResult(
                conversation_id=str(uuid.uuid4()),
                objective=kwargs["objective"],
                attack_identifier={"__type__": "TestAttack"},
                outcome=AttackOutcome.SUCCESS,
                executed_turns=1,
            )

        mock_attack_strategy.execute_async.side_effect = mock_execute

        objectives = ["Obj1", "Obj2", "Obj3"]

        result = await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
            return_partial_on_failure=True,
        )

        assert len(result.completed_results) == 2
        assert len(result.incomplete_objectives) == 1
        assert result.has_incomplete
        assert result.incomplete_objectives[0][0] == "Obj2"

    @pytest.mark.asyncio
    async def test_partial_failure_raises_by_default(self, mock_attack_strategy):
        """Test that failures raise exception by default"""
        executor = AttackExecutor(max_concurrency=3)

        async def mock_execute(**kwargs):
            if kwargs["objective"] == "Obj2":
                raise RuntimeError("Execution failed")
            return MagicMock()

        mock_attack_strategy.execute_async.side_effect = mock_execute

        objectives = ["Obj1", "Obj2", "Obj3"]

        with pytest.raises(RuntimeError, match="Execution failed"):
            await executor.execute_attack_async(
                attack=mock_attack_strategy,
                objectives=objectives,
            )

    @pytest.mark.asyncio
    async def test_preserves_result_order(self, mock_attack_strategy):
        """Test that results maintain objective order despite varying execution times"""
        executor = AttackExecutor(max_concurrency=3)
        objectives = [f"Objective {i}" for i in range(5)]

        async def mock_execute(**kwargs):
            # Vary execution times
            idx = int(kwargs["objective"].split()[-1])
            delay = 0.1 if idx in [1, 3] else 0.01
            await asyncio.sleep(delay)
            return AttackResult(
                conversation_id=str(uuid.uuid4()),
                objective=kwargs["objective"],
                attack_identifier={"__type__": "TestAttack"},
                outcome=AttackOutcome.SUCCESS,
                executed_turns=1,
            )

        mock_attack_strategy.execute_async.side_effect = mock_execute

        results = await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
        )

        for i, result in enumerate(results):
            assert result.objective == objectives[i]


@pytest.mark.usefixtures("patch_central_database")
class TestExecuteAttackFromSeedGroupsAsync:
    """Tests for execute_attack_from_seed_groups_async method"""

    @pytest.fixture
    def sample_seed_groups(self):
        """Create sample seed groups for testing"""
        from pyrit.models import SeedGroup, SeedObjective, SeedPrompt

        seed_groups = []
        for i in range(3):
            objective = SeedObjective(value=f"Objective {i}")
            next_msg = SeedPrompt(value=f"Next message {i}", data_type="text")
            seed_group = SeedGroup(seeds=[objective, next_msg])
            seed_groups.append(seed_group)
        return seed_groups

    @pytest.mark.asyncio
    async def test_extracts_objectives_from_seed_groups(self, mock_attack_strategy, sample_seed_groups):
        """Test that objectives are extracted from seed groups"""
        executor = AttackExecutor(max_concurrency=3)
        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_from_seed_groups_async(
            attack=mock_attack_strategy,
            seed_groups=sample_seed_groups,
        )

        calls = mock_attack_strategy.execute_async.call_args_list
        for i, call in enumerate(calls):
            assert call.kwargs["objective"] == f"Objective {i}"

    @pytest.mark.asyncio
    async def test_extracts_next_messages_from_seed_groups(self, mock_attack_strategy, sample_seed_groups):
        """Test that next_messages are extracted from seed groups"""
        executor = AttackExecutor(max_concurrency=3)
        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_from_seed_groups_async(
            attack=mock_attack_strategy,
            seed_groups=sample_seed_groups,
        )

        calls = mock_attack_strategy.execute_async.call_args_list
        for i, call in enumerate(calls):
            # next_message should be extracted from the seed group
            assert "next_message" in call.kwargs

    @pytest.mark.asyncio
    async def test_validates_empty_seed_groups(self, mock_attack_strategy):
        """Test that empty seed_groups raises ValueError"""
        executor = AttackExecutor()

        with pytest.raises(ValueError, match="At least one seed_group must be provided"):
            await executor.execute_attack_from_seed_groups_async(
                attack=mock_attack_strategy,
                seed_groups=[],
            )

    @pytest.mark.asyncio
    async def test_validates_seed_group_has_objective(self, mock_attack_strategy):
        """Test that seed groups without objectives raise ValueError"""
        from pyrit.models import SeedGroup, SeedPrompt

        executor = AttackExecutor()

        # Create seed group without objective (only SeedPrompt, no SeedObjective)
        prompt = SeedPrompt(value="Not an objective", data_type="text")
        seed_group = SeedGroup(seeds=[prompt])

        with pytest.raises(ValueError, match="does not have an objective"):
            await executor.execute_attack_from_seed_groups_async(
                attack=mock_attack_strategy,
                seed_groups=[seed_group],
            )

    @pytest.mark.asyncio
    async def test_passes_memory_labels(self, mock_attack_strategy, sample_seed_groups):
        """Test that memory_labels are passed through"""
        executor = AttackExecutor(max_concurrency=3)
        mock_attack_strategy.execute_async.return_value = MagicMock()
        memory_labels = {"test": "label"}

        await executor.execute_attack_from_seed_groups_async(
            attack=mock_attack_strategy,
            seed_groups=sample_seed_groups,
            memory_labels=memory_labels,
        )

        for call in mock_attack_strategy.execute_async.call_args_list:
            assert call.kwargs["memory_labels"] == memory_labels

    @pytest.mark.asyncio
    async def test_passes_broadcast_attack_params(self, mock_attack_strategy, sample_seed_groups):
        """Test that broadcast attack params are passed through"""
        executor = AttackExecutor(max_concurrency=3)
        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_from_seed_groups_async(
            attack=mock_attack_strategy,
            seed_groups=sample_seed_groups,
            max_turns=5,
        )

        for call in mock_attack_strategy.execute_async.call_args_list:
            assert call.kwargs["max_turns"] == 5


@pytest.mark.usefixtures("patch_central_database")
class TestDeprecatedMethodsEmitWarnings:
    """Tests that deprecated methods emit deprecation warnings and still work"""

    @pytest.mark.asyncio
    async def test_execute_multi_objective_attack_async_emits_warning(self, mock_attack_strategy, caplog):
        """Test that execute_multi_objective_attack_async emits deprecation warning"""
        import logging

        executor = AttackExecutor(max_concurrency=1)
        mock_attack_strategy.execute_async.return_value = MagicMock()

        with caplog.at_level(logging.WARNING):
            await executor.execute_multi_objective_attack_async(
                attack=mock_attack_strategy,
                objectives=["Test objective"],
            )

        assert "execute_multi_objective_attack_async is deprecated" in caplog.text

    @pytest.mark.asyncio
    async def test_execute_single_turn_attacks_async_emits_warning(self, mock_single_turn_attack_strategy, caplog):
        """Test that execute_single_turn_attacks_async emits deprecation warning"""
        import logging

        executor = AttackExecutor(max_concurrency=1)
        mock_single_turn_attack_strategy.execute_async.return_value = MagicMock()

        with caplog.at_level(logging.WARNING):
            await executor.execute_single_turn_attacks_async(
                attack=mock_single_turn_attack_strategy,
                objectives=["Test objective"],
            )

        assert "execute_single_turn_attacks_async is deprecated" in caplog.text

    @pytest.mark.asyncio
    async def test_execute_multi_turn_attacks_async_emits_warning(self, mock_multi_turn_attack_strategy, caplog):
        """Test that execute_multi_turn_attacks_async emits deprecation warning"""
        import logging

        executor = AttackExecutor(max_concurrency=1)
        mock_multi_turn_attack_strategy.execute_async.return_value = MagicMock()

        with caplog.at_level(logging.WARNING):
            await executor.execute_multi_turn_attacks_async(
                attack=mock_multi_turn_attack_strategy,
                objectives=["Test objective"],
            )

        assert "execute_multi_turn_attacks_async is deprecated" in caplog.text

    @pytest.mark.asyncio
    async def test_execute_multi_objective_attack_with_context_async_emits_warning(
        self, mock_attack_strategy, basic_context, caplog
    ):
        """Test that execute_multi_objective_attack_with_context_async emits deprecation warning"""
        import logging

        executor = AttackExecutor(max_concurrency=1)
        basic_context.duplicate = MagicMock(return_value=MagicMock(objective=None))
        mock_attack_strategy.execute_with_context_async.return_value = MagicMock()

        with caplog.at_level(logging.WARNING):
            await executor.execute_multi_objective_attack_with_context_async(
                attack=mock_attack_strategy,
                context_template=basic_context,
                objectives=["Test objective"],
            )

        assert "execute_multi_objective_attack_with_context_async is deprecated" in caplog.text


@pytest.mark.usefixtures("patch_central_database")
class TestConcurrencyControl:
    """Tests for concurrency control in attack execution using execute_attack_async"""

    @pytest.mark.asyncio
    async def test_respects_max_concurrency_limit(self, mock_attack_strategy):
        max_concurrency = 2
        executor = AttackExecutor(max_concurrency=max_concurrency)

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0

        async def mock_execute(**kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)
            concurrent_count -= 1
            return MagicMock()

        mock_attack_strategy.execute_async.side_effect = mock_execute

        objectives = [f"Objective {i}" for i in range(10)]

        await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
        )

        assert max_concurrent <= max_concurrency
        assert mock_attack_strategy.execute_async.call_count == len(objectives)

    @pytest.mark.asyncio
    async def test_single_concurrency_serializes_execution(self, mock_attack_strategy):
        executor = AttackExecutor(max_concurrency=1)

        execution_order = []

        async def mock_execute(**kwargs):
            objective = kwargs["objective"]
            execution_order.append(f"start_{objective}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end_{objective}")
            return MagicMock()

        mock_attack_strategy.execute_async.side_effect = mock_execute

        objectives = ["A", "B", "C"]

        await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
        )

        # With max_concurrency=1, executions should not overlap
        expected_order = ["start_A", "end_A", "start_B", "end_B", "start_C", "end_C"]
        assert execution_order == expected_order


@pytest.mark.usefixtures("patch_central_database")
class TestErrorHandling:
    """Tests for error handling in attack execution using execute_attack_async"""

    @pytest.mark.asyncio
    async def test_propagates_strategy_execution_errors(self, mock_attack_strategy):
        executor = AttackExecutor(max_concurrency=5)

        mock_attack_strategy.execute_async.side_effect = RuntimeError("Strategy execution failed")

        with pytest.raises(RuntimeError, match="Strategy execution failed"):
            await executor.execute_attack_async(
                attack=mock_attack_strategy,
                objectives=["Test objective"],
            )

    @pytest.mark.asyncio
    async def test_handles_partial_failures(self, mock_attack_strategy):
        executor = AttackExecutor(max_concurrency=5)

        # Define execution behavior based on objective
        async def mock_execute(**kwargs):
            if kwargs["objective"] == "Failure":
                raise RuntimeError("Execution failed")
            return AttackResult(
                conversation_id=str(uuid.uuid4()),
                objective=kwargs["objective"],
                attack_identifier={"__type__": "TestAttack"},
                outcome=AttackOutcome.SUCCESS,
                executed_turns=1,
            )

        mock_attack_strategy.execute_async.side_effect = mock_execute

        objectives = ["Success 1", "Failure", "Success 2"]

        # Test default behavior (return_partial_on_failure=False): should raise
        with pytest.raises(RuntimeError, match="Execution failed"):
            await executor.execute_attack_async(
                attack=mock_attack_strategy,
                objectives=objectives,
            )

        # Reset the mock
        mock_attack_strategy.execute_async.reset_mock()
        mock_attack_strategy.execute_async.side_effect = mock_execute

        # Test with return_partial_on_failure=True: should return partial results
        result = await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
            return_partial_on_failure=True,
        )

        # Verify we got partial results
        assert len(result.completed_results) == 2
        assert len(result.incomplete_objectives) == 1
        assert result.has_incomplete
        assert not result.all_completed

        # Verify the incomplete objective details
        failed_objective, exception = result.incomplete_objectives[0]
        assert failed_objective == "Failure"
        assert isinstance(exception, RuntimeError)
        assert str(exception) == "Execution failed"

        # Verify completed results
        assert result.completed_results[0].objective == "Success 1"
        assert result.completed_results[1].objective == "Success 2"


@pytest.mark.usefixtures("patch_central_database")
class TestIntegrationScenarios:
    """Tests for integration scenarios and edge cases using execute_attack_async"""

    @pytest.mark.asyncio
    async def test_large_scale_execution(self, mock_attack_strategy):
        executor = AttackExecutor(max_concurrency=10)

        # Test with a large number of objectives
        objectives = [f"Objective {i}" for i in range(100)]

        async def mock_execute(**kwargs):
            await asyncio.sleep(0.001)
            return AttackResult(
                conversation_id=str(uuid.uuid4()),
                objective=kwargs["objective"],
                attack_identifier={"__type__": "TestAttack"},
                outcome=AttackOutcome.SUCCESS,
                executed_turns=1,
            )

        mock_attack_strategy.execute_async.side_effect = mock_execute

        results = await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
        )

        assert len(results) == 100
        assert all(isinstance(r, AttackResult) for r in results)
        assert mock_attack_strategy.execute_async.call_count == 100

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure_results(self, mock_attack_strategy):
        executor = AttackExecutor(max_concurrency=3)

        objectives = [f"Objective {i}" for i in range(6)]

        async def mock_execute(**kwargs):
            idx = int(kwargs["objective"].split()[-1])
            return AttackResult(
                conversation_id=str(uuid.uuid4()),
                objective=kwargs["objective"],
                attack_identifier={"__type__": "TestAttack"},
                outcome=AttackOutcome.SUCCESS if idx % 2 == 0 else AttackOutcome.FAILURE,
                executed_turns=idx + 1,
            )

        mock_attack_strategy.execute_async.side_effect = mock_execute

        results = await executor.execute_attack_async(
            attack=mock_attack_strategy,
            objectives=objectives,
        )

        # Verify alternating pattern
        for i, result in enumerate(results):
            assert result.outcome == (AttackOutcome.SUCCESS if i % 2 == 0 else AttackOutcome.FAILURE)
            assert result.executed_turns == i + 1


@pytest.mark.usefixtures("patch_central_database")
class TestValidateAttackBatchParameters:
    """Tests for the shared validation method"""

    def test_validate_empty_objectives(self):
        executor = AttackExecutor()

        with pytest.raises(ValueError, match="At least one objective must be provided"):
            executor._validate_attack_batch_parameters(objectives=[])

    def test_validate_optional_list_length_mismatch(self):
        executor = AttackExecutor()
        objectives = ["Obj1", "Obj2"]
        optional_list = ["Only one item"]

        with pytest.raises(ValueError, match="Number of test_list .* must match number of objectives"):
            executor._validate_attack_batch_parameters(
                objectives=objectives, optional_list=optional_list, optional_list_name="test_list"
            )

    def test_validate_prepended_conversations_length_mismatch(self):
        executor = AttackExecutor()
        objectives = ["Obj1", "Obj2"]
        prepended_conversations = [[]]  # Only one conversation

        with pytest.raises(ValueError, match="Number of prepended_conversations .* must match"):
            executor._validate_attack_batch_parameters(
                objectives=objectives, prepended_conversations=prepended_conversations
            )

    def test_validate_success_with_valid_parameters(self):
        executor = AttackExecutor()
        objectives = ["Obj1", "Obj2"]
        optional_list = ["Item1", "Item2"]
        prepended_conversations = [[], []]

        # Should not raise any exception
        executor._validate_attack_batch_parameters(
            objectives=objectives,
            optional_list=optional_list,
            optional_list_name="test_list",
            prepended_conversations=prepended_conversations,
        )

    def test_validate_success_with_none_optional_parameters(self):
        executor = AttackExecutor()
        objectives = ["Obj1", "Obj2"]

        # Should not raise any exception when optional parameters are None
        executor._validate_attack_batch_parameters(
            objectives=objectives, optional_list=None, prepended_conversations=None
        )


@pytest.mark.usefixtures("patch_central_database")
class TestBuildPerAttackParamsFromSeedGroups:
    """Tests for _build_per_attack_params_from_seed_groups method"""

    @pytest.fixture
    def seed_groups_without_messages(self):
        """Create seed groups that have no multi-turn messages (only objective + next_message)."""
        from pyrit.models import SeedGroup, SeedObjective, SeedPrompt

        seed_groups = []
        for i in range(2):
            objective = SeedObjective(value=f"Objective {i}")
            # Single next message prompt - not a multi-turn sequence
            next_msg = SeedPrompt(value=f"Next message {i}", data_type="text", sequence=0)
            seed_group = SeedGroup(seeds=[objective, next_msg])
            seed_groups.append(seed_group)
        return seed_groups

    @pytest.fixture
    def seed_groups_with_messages(self):
        """Create seed groups with multi-turn message sequences."""
        from pyrit.models import SeedGroup, SeedObjective, SeedPrompt

        seed_groups = []
        for i in range(2):
            objective = SeedObjective(value=f"Objective {i}")
            # Multi-turn sequence with multiple prompts - roles required for multi-sequence
            prompts = [
                SeedPrompt(value=f"First message {i}", data_type="text", sequence=0, role="user"),
                SeedPrompt(value=f"Second message {i}", data_type="text", sequence=1, role="user"),
                SeedPrompt(value=f"Third message {i}", data_type="text", sequence=2, role="user"),
            ]
            seed_group = SeedGroup(seeds=[objective] + prompts)
            seed_groups.append(seed_group)
        return seed_groups

    @pytest.fixture
    def mixed_seed_groups(self):
        """Create seed groups where only some have messages."""
        from pyrit.models import SeedGroup, SeedObjective, SeedPrompt

        # First group without multi-turn messages
        objective1 = SeedObjective(value="Objective 0")
        sg1 = SeedGroup(seeds=[objective1])

        # Second group with multi-turn messages - roles required for multi-sequence
        objective2 = SeedObjective(value="Objective 1")
        prompts = [
            SeedPrompt(value="Message 1", data_type="text", sequence=0, role="user"),
            SeedPrompt(value="Message 2", data_type="text", sequence=1, role="user"),
        ]
        sg2 = SeedGroup(seeds=[objective2] + prompts)

        return [sg1, sg2]

    def test_returns_none_when_no_messages_and_no_per_attack_params(self, seed_groups_without_messages):
        """Test that None is returned when no seed group has messages and no per_attack_params."""
        executor = AttackExecutor()

        # Seed groups with only objective - messages property returns empty list
        from pyrit.models import SeedGroup, SeedObjective

        simple_groups = [SeedGroup(seeds=[SeedObjective(value="obj")]) for _ in range(2)]

        result = executor._build_per_attack_params_from_seed_groups(
            seed_groups=simple_groups,
            per_attack_params=None,
        )

        assert result is None

    def test_returns_params_when_seed_groups_have_messages(self, seed_groups_with_messages):
        """Test that messages are extracted from seed groups."""
        executor = AttackExecutor()

        result = executor._build_per_attack_params_from_seed_groups(
            seed_groups=seed_groups_with_messages,
            per_attack_params=None,
        )

        assert result is not None
        assert len(result) == 2

        for i, params in enumerate(result):
            assert "messages" in params
            # Messages should be a list of Message objects
            assert len(params["messages"]) == 3
            # Verify message contents
            assert params["messages"][0].message_pieces[0].original_value == f"First message {i}"

    def test_merges_with_existing_per_attack_params(self, seed_groups_with_messages):
        """Test that extracted messages merge with user-provided per_attack_params."""
        executor = AttackExecutor()
        per_attack_params = [
            {"max_turns": 5},
            {"max_turns": 10, "custom_param": "value"},
        ]

        result = executor._build_per_attack_params_from_seed_groups(
            seed_groups=seed_groups_with_messages,
            per_attack_params=per_attack_params,
        )

        assert result is not None
        assert len(result) == 2

        # First group: messages + max_turns
        assert "messages" in result[0]
        assert result[0]["max_turns"] == 5

        # Second group: messages + max_turns + custom_param
        assert "messages" in result[1]
        assert result[1]["max_turns"] == 10
        assert result[1]["custom_param"] == "value"

    def test_user_provided_messages_take_precedence(self, seed_groups_with_messages):
        """Test that user-provided messages override extracted messages."""
        executor = AttackExecutor()
        from pyrit.models import Message

        custom_messages = [Message.from_prompt(prompt="Custom message", role="user")]
        per_attack_params = [
            {"messages": custom_messages},  # User provides their own messages
            {},  # No override, use extracted
        ]

        result = executor._build_per_attack_params_from_seed_groups(
            seed_groups=seed_groups_with_messages,
            per_attack_params=per_attack_params,
        )

        assert result is not None

        # First group: user-provided messages take precedence
        assert result[0]["messages"] == custom_messages

        # Second group: extracted messages from seed group
        assert len(result[1]["messages"]) == 3

    def test_handles_mixed_seed_groups(self, mixed_seed_groups):
        """Test handling of seed groups where only some have messages."""
        executor = AttackExecutor()

        result = executor._build_per_attack_params_from_seed_groups(
            seed_groups=mixed_seed_groups,
            per_attack_params=None,
        )

        assert result is not None
        assert len(result) == 2

        # First group has no messages (empty dict or no messages key)
        assert "messages" not in result[0] or result[0].get("messages") is None

        # Second group has messages
        assert "messages" in result[1]
        assert len(result[1]["messages"]) == 2

    def test_returns_per_attack_params_when_no_messages(self):
        """Test that per_attack_params are returned even when no messages exist."""
        executor = AttackExecutor()
        from pyrit.models import SeedGroup, SeedObjective

        # Simple groups without messages
        simple_groups = [SeedGroup(seeds=[SeedObjective(value="obj")]) for _ in range(2)]
        per_attack_params = [{"custom": "value1"}, {"custom": "value2"}]

        result = executor._build_per_attack_params_from_seed_groups(
            seed_groups=simple_groups,
            per_attack_params=per_attack_params,
        )

        assert result is not None
        assert len(result) == 2
        assert result[0]["custom"] == "value1"
        assert result[1]["custom"] == "value2"


@pytest.mark.usefixtures("patch_central_database")
class TestExecuteAttackFromSeedGroupsWithMessages:
    """Tests for execute_attack_from_seed_groups_async with messages extraction"""

    @pytest.fixture
    def seed_groups_with_messages(self):
        """Create seed groups with multi-turn message sequences."""
        from pyrit.models import SeedGroup, SeedObjective, SeedPrompt

        seed_groups = []
        for i in range(2):
            objective = SeedObjective(value=f"Objective {i}")
            # Roles required for multi-sequence groups
            prompts = [
                SeedPrompt(value=f"Message {i}-0", data_type="text", sequence=0, role="user"),
                SeedPrompt(value=f"Message {i}-1", data_type="text", sequence=1, role="user"),
            ]
            seed_group = SeedGroup(seeds=[objective] + prompts)
            seed_groups.append(seed_group)
        return seed_groups

    @pytest.mark.asyncio
    async def test_messages_passed_to_attack(self, mock_attack_strategy, seed_groups_with_messages):
        """Test that messages from seed groups are passed to the attack."""
        executor = AttackExecutor(max_concurrency=2)
        mock_attack_strategy.execute_async.return_value = MagicMock()

        await executor.execute_attack_from_seed_groups_async(
            attack=mock_attack_strategy,
            seed_groups=seed_groups_with_messages,
        )

        calls = mock_attack_strategy.execute_async.call_args_list
        assert len(calls) == 2

        # Verify each call received the messages
        for i, call in enumerate(calls):
            assert "messages" in call.kwargs
            messages = call.kwargs["messages"]
            assert len(messages) == 2
            assert messages[0].message_pieces[0].original_value == f"Message {i}-0"
            assert messages[1].message_pieces[0].original_value == f"Message {i}-1"

    @pytest.mark.asyncio
    async def test_messages_filtered_for_unsupported_attacks(self):
        """Test that messages are filtered out for attacks that don't accept them."""
        from pyrit.models import SeedGroup, SeedObjective, SeedPrompt

        # Create seed group with messages (multi-sequence requires roles)
        seed_group = SeedGroup(
            seeds=[
                SeedObjective(value="Test objective"),
                SeedPrompt(value="Message 1", data_type="text", sequence=0, role="user"),
                SeedPrompt(value="Message 2", data_type="text", sequence=1, role="user"),
            ]
        )

        executor = AttackExecutor(max_concurrency=1)

        # Create a mock attack that does NOT accept 'messages' parameter
        mock_attack = MagicMock(spec=AttackStrategy)
        mock_attack.accepted_context_parameters = SINGLE_TURN_ACCEPTED_PARAMS  # No 'messages'
        mock_attack.execute_async = AsyncMock(return_value=MagicMock())

        await executor.execute_attack_from_seed_groups_async(
            attack=mock_attack,
            seed_groups=[seed_group],
        )

        # Verify messages were NOT passed to the attack (filtered out)
        call_kwargs = mock_attack.execute_async.call_args.kwargs
        assert "messages" not in call_kwargs
        assert "objective" in call_kwargs


@pytest.mark.usefixtures("patch_central_database")
class TestFilterParamsForAttack:
    """Tests for _filter_params_for_attack method and accepted_context_parameters property."""

    def test_filter_params_keeps_accepted_parameters(self):
        """Test that parameters accepted by the context type are kept."""
        executor = AttackExecutor(max_concurrency=1)

        # Create a mock attack with SingleTurnAttackContext (has objective, memory_labels, etc.)
        mock_attack = MagicMock(spec=AttackStrategy)
        mock_attack.accepted_context_parameters = {"objective", "memory_labels", "next_message"}

        params = {
            "objective": "Test objective",
            "memory_labels": {"key": "value"},
            "next_message": MagicMock(),
        }

        filtered = executor._filter_params_for_attack(
            attack=mock_attack,
            params=params,
            strict_param_matching=False,
        )

        assert filtered == params

    def test_filter_params_removes_unsupported_parameters(self):
        """Test that parameters not accepted by the context type are filtered out."""
        executor = AttackExecutor(max_concurrency=1)

        # Create a mock attack with limited accepted parameters (like SingleTurnAttackContext)
        mock_attack = MagicMock(spec=AttackStrategy)
        mock_attack.accepted_context_parameters = {"objective", "memory_labels", "next_message"}

        params = {
            "objective": "Test objective",
            "memory_labels": {"key": "value"},
            "messages": [MagicMock()],  # Not in accepted parameters
            "unsupported_param": "should be filtered",
        }

        filtered = executor._filter_params_for_attack(
            attack=mock_attack,
            params=params,
            strict_param_matching=False,
        )

        assert "objective" in filtered
        assert "memory_labels" in filtered
        assert "messages" not in filtered
        assert "unsupported_param" not in filtered

    def test_filter_params_strict_mode_raises_for_unsupported(self):
        """Test that strict_param_matching=True raises ValueError for unsupported params."""
        executor = AttackExecutor(max_concurrency=1)

        mock_attack = MagicMock(spec=AttackStrategy)
        mock_attack.accepted_context_parameters = {"objective", "memory_labels"}

        params = {
            "objective": "Test objective",
            "unsupported_param": "should cause error",
        }

        with pytest.raises(ValueError, match="does not accept parameters"):
            executor._filter_params_for_attack(
                attack=mock_attack,
                params=params,
                strict_param_matching=True,
            )

    def test_filter_params_ignores_none_values(self):
        """Test that None values are not included in filtered params."""
        executor = AttackExecutor(max_concurrency=1)

        mock_attack = MagicMock(spec=AttackStrategy)
        mock_attack.accepted_context_parameters = {"objective", "memory_labels", "next_message"}

        params = {
            "objective": "Test objective",
            "memory_labels": None,  # Should be ignored
            "next_message": None,  # Should be ignored
        }

        filtered = executor._filter_params_for_attack(
            attack=mock_attack,
            params=params,
            strict_param_matching=False,
        )

        assert "objective" in filtered
        assert "memory_labels" not in filtered
        assert "next_message" not in filtered


@pytest.mark.usefixtures("patch_central_database")
class TestStrategyContextTypeValidation:
    """Tests for Strategy context_type validation and accepted_context_parameters property."""

    def test_strategy_init_with_non_dataclass_context_raises_type_error(self):
        """Test that Strategy raises TypeError when context_type is not a dataclass."""
        from pyrit.executor.core.strategy import Strategy

        # Create a non-dataclass context (not inheriting from StrategyContext which is a dataclass)
        class InvalidContext:
            def __init__(self, objective: str):
                self.objective = objective

        # Create a concrete strategy implementation for testing
        class TestStrategy(Strategy):
            def _validate_context(self, *, context):
                pass

            async def _setup_async(self, *, context):
                pass

            async def _perform_async(self, *, context):
                return None

            async def _teardown_async(self, *, context):
                pass

        with pytest.raises(TypeError, match="context_type must be a dataclass"):
            TestStrategy(context_type=InvalidContext)

    def test_strategy_init_error_message_includes_class_name(self):
        """Test that the error message includes the invalid class name."""
        from pyrit.executor.core.strategy import Strategy

        # Create a non-dataclass context (not inheriting from StrategyContext which is a dataclass)
        class MyInvalidContext:
            pass

        class TestStrategy(Strategy):
            def _validate_context(self, *, context):
                pass

            async def _setup_async(self, *, context):
                pass

            async def _perform_async(self, *, context):
                return None

            async def _teardown_async(self, *, context):
                pass

        with pytest.raises(TypeError, match="MyInvalidContext"):
            TestStrategy(context_type=MyInvalidContext)

    def test_accepted_context_parameters_returns_dataclass_fields(self):
        """Test that accepted_context_parameters returns all dataclass fields."""
        from dataclasses import dataclass

        from pyrit.executor.core.strategy import Strategy, StrategyContext

        @dataclass
        class ValidContext(StrategyContext):
            objective: str
            custom_field: str = "default"

        class TestStrategy(Strategy):
            def _validate_context(self, *, context):
                pass

            async def _setup_async(self, *, context):
                pass

            async def _perform_async(self, *, context):
                return None

            async def _teardown_async(self, *, context):
                pass

        strategy = TestStrategy(context_type=ValidContext)
        accepted = strategy.accepted_context_parameters

        assert "objective" in accepted
        assert "custom_field" in accepted

    def test_accepted_context_parameters_includes_inherited_fields(self):
        """Test that accepted_context_parameters includes fields from parent dataclasses."""
        from pyrit.executor.core.strategy import Strategy

        class TestStrategy(Strategy):
            def _validate_context(self, *, context):
                pass

            async def _setup_async(self, *, context):
                pass

            async def _perform_async(self, *, context):
                return None

            async def _teardown_async(self, *, context):
                pass

        # SingleTurnAttackContext inherits from AttackContext
        strategy = TestStrategy(context_type=SingleTurnAttackContext)
        accepted = strategy.accepted_context_parameters

        # From SingleTurnAttackContext
        assert "next_message" in accepted
        assert "conversation_id" in accepted
        # From AttackContext (parent)
        assert "objective" in accepted
        assert "memory_labels" in accepted
        assert "prepended_conversation" in accepted


@pytest.mark.usefixtures("patch_central_database")
class TestExcludedContextParameters:
    """Tests for _excluded_context_parameters integration with accepted_context_parameters."""

    def test_accepted_context_parameters_excludes_strategy_excluded_parameters(self):
        """Test that accepted_context_parameters automatically excludes _excluded_context_parameters."""
        from pyrit.executor.core.strategy import Strategy

        class TestStrategy(Strategy):
            def _validate_context(self, *, context):
                pass

            async def _setup_async(self, *, context):
                pass

            async def _perform_async(self, *, context):
                return None

            async def _teardown_async(self, *, context):
                pass

            @property
            def _excluded_context_parameters(self) -> frozenset[str]:
                return frozenset({"next_message", "prepended_conversation"})

        strategy = TestStrategy(context_type=SingleTurnAttackContext)
        accepted = strategy.accepted_context_parameters

        # These should be excluded
        assert "next_message" not in accepted
        assert "prepended_conversation" not in accepted
        # These should still be accepted
        assert "objective" in accepted
        assert "memory_labels" in accepted

    def test_filter_params_excludes_strategy_excluded_parameters(self):
        """Test that parameters excluded by the strategy are filtered out by AttackExecutor."""
        from pyrit.executor.core.strategy import Strategy

        class TestStrategy(Strategy):
            def _validate_context(self, *, context):
                pass

            async def _setup_async(self, *, context):
                pass

            async def _perform_async(self, *, context):
                return None

            async def _teardown_async(self, *, context):
                pass

            @property
            def _excluded_context_parameters(self) -> frozenset[str]:
                return frozenset({"next_message", "prepended_conversation"})

        executor = AttackExecutor(max_concurrency=1)
        strategy = TestStrategy(context_type=SingleTurnAttackContext)

        params = {
            "objective": "Test objective",
            "memory_labels": {"key": "value"},
            "next_message": MagicMock(),  # Should be excluded
            "prepended_conversation": [MagicMock()],  # Should be excluded
        }

        filtered = executor._filter_params_for_attack(
            attack=strategy,
            params=params,
            strict_param_matching=False,
        )

        assert "objective" in filtered
        assert "memory_labels" in filtered
        assert "next_message" not in filtered
        assert "prepended_conversation" not in filtered

    def test_filter_params_empty_excluded_set_does_not_affect_filtering(self):
        """Test that an empty _excluded_context_parameters doesn't affect normal filtering."""
        executor = AttackExecutor(max_concurrency=1)

        mock_attack = MagicMock(spec=AttackStrategy)
        mock_attack.accepted_context_parameters = {"objective", "memory_labels", "next_message"}

        params = {
            "objective": "Test objective",
            "next_message": MagicMock(),
        }

        filtered = executor._filter_params_for_attack(
            attack=mock_attack,
            params=params,
            strict_param_matching=False,
        )

        assert "objective" in filtered
        assert "next_message" in filtered

    def test_strategy_base_excluded_context_parameters_returns_empty_frozenset(self):
        """Test that the base Strategy class returns an empty frozenset for _excluded_context_parameters."""
        from pyrit.executor.core.strategy import Strategy

        class TestStrategy(Strategy):
            def _validate_context(self, *, context):
                pass

            async def _setup_async(self, *, context):
                pass

            async def _perform_async(self, *, context):
                return None

            async def _teardown_async(self, *, context):
                pass

        strategy = TestStrategy(context_type=SingleTurnAttackContext)
        excluded = strategy._excluded_context_parameters

        assert excluded == frozenset()
        assert isinstance(excluded, frozenset)

    def test_strategy_accepted_parameters_includes_all_fields_when_no_exclusions(self):
        """Test that accepted_context_parameters includes all dataclass fields when nothing is excluded."""
        from pyrit.executor.core.strategy import Strategy

        class TestStrategy(Strategy):
            def _validate_context(self, *, context):
                pass

            async def _setup_async(self, *, context):
                pass

            async def _perform_async(self, *, context):
                return None

            async def _teardown_async(self, *, context):
                pass

        strategy = TestStrategy(context_type=SingleTurnAttackContext)
        accepted = strategy.accepted_context_parameters

        # All SingleTurnAttackContext fields should be present
        assert "next_message" in accepted
        assert "prepended_conversation" in accepted
        assert "objective" in accepted
        assert "memory_labels" in accepted
