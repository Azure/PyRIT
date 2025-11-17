# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the scenarios.AtomicAttack class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import AttackExecutor, AttackStrategy
from pyrit.executor.attack.core import AttackExecutorResult
from pyrit.models import AttackOutcome, AttackResult, Message
from pyrit.scenarios import AtomicAttack


@pytest.fixture
def mock_attack():
    """Create a mock AttackStrategy for testing."""
    return MagicMock(spec=AttackStrategy)


@pytest.fixture
def sample_objectives():
    """Create sample objectives for testing."""
    return ["objective1", "objective2", "objective3"]


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


def wrap_results(results):
    """Helper to wrap attack results in AttackExecutorResult."""
    return AttackExecutorResult(completed_results=results, incomplete_objectives=[])


@pytest.fixture
def sample_conversation():
    """Create sample conversation for testing."""
    return [
        MagicMock(spec=Message),
        MagicMock(spec=Message),
    ]


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackInitialization:
    """Tests for AtomicAttack class initialization."""

    def test_init_with_valid_params(self, mock_attack, sample_objectives):
        """Test successful initialization with valid parameters."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            atomic_attack_name="Test Attack Run",
        )

        assert atomic_attack._attack == mock_attack
        assert atomic_attack._objectives == sample_objectives
        assert atomic_attack._prepended_conversations is None
        assert atomic_attack._memory_labels == {}
        assert atomic_attack._attack_execute_params == {}

    def test_init_with_memory_labels(self, mock_attack, sample_objectives):
        """Test initialization with memory labels."""
        memory_labels = {"test": "label", "category": "attack"}

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            memory_labels=memory_labels,
            atomic_attack_name="Test Attack Run",
        )

        assert atomic_attack._memory_labels == memory_labels

    def test_init_with_prepended_conversation(self, mock_attack, sample_objectives, sample_conversation):
        """Test initialization with prepended conversation."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            prepended_conversations=sample_conversation,
            atomic_attack_name="Test Attack Run",
        )

        assert atomic_attack._prepended_conversations == sample_conversation

    def test_init_with_attack_execute_params(self, mock_attack, sample_objectives):
        """Test initialization with additional attack execute parameters."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            max_retries=5,
            custom_param="value",
            atomic_attack_name="Test Attack Run",
        )

        assert atomic_attack._attack_execute_params["max_retries"] == 5
        assert atomic_attack._attack_execute_params["custom_param"] == "value"

    def test_init_with_all_parameters(self, mock_attack, sample_objectives, sample_conversation):
        """Test initialization with all parameters."""
        memory_labels = {"test": "comprehensive"}

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            prepended_conversations=sample_conversation,
            memory_labels=memory_labels,
            batch_size=10,
            timeout=30,
            atomic_attack_name="Test Attack Run",
        )

        assert atomic_attack._attack == mock_attack
        assert atomic_attack._objectives == sample_objectives
        assert atomic_attack._prepended_conversations == sample_conversation
        assert atomic_attack._memory_labels == memory_labels
        assert atomic_attack._attack_execute_params["batch_size"] == 10
        assert atomic_attack._attack_execute_params["timeout"] == 30

    def test_init_fails_with_empty_objectives(self, mock_attack):
        """Test that initialization fails when objectives list is empty."""
        with pytest.raises(ValueError, match="objectives list cannot be empty"):
            AtomicAttack(
                attack=mock_attack,
                objectives=[],
                atomic_attack_name="Test Attack Run",
            )


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackExecution:
    """Tests for AtomicAttack execution methods."""

    @pytest.mark.asyncio
    async def test_run_async_with_valid_atomic_attack(self, mock_attack, sample_objectives, sample_attack_results):
        """Test successful execution of an atomic attack."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            atomic_attack_name="Test Attack Run",
        )

        # Mock the executor
        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            result = await atomic_attack.run_async()

            assert len(result.completed_results) == 3
            assert result.completed_results == sample_attack_results
            assert len(result.incomplete_objectives) == 0
            mock_exec.assert_called_once()

            # Verify the attack was passed correctly
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == mock_attack

    @pytest.mark.asyncio
    async def test_run_async_with_custom_concurrency(self, mock_attack, sample_objectives, sample_attack_results):
        """Test execution with custom max_concurrency for atomic attack."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "__init__", return_value=None) as mock_init:
            with patch.object(
                AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = wrap_results(sample_attack_results)

                result = await atomic_attack.run_async(max_concurrency=5)

                mock_init.assert_called_once_with(max_concurrency=5)
                assert len(result.completed_results) == 3

    @pytest.mark.asyncio
    async def test_run_async_with_default_concurrency(self, mock_attack, sample_objectives, sample_attack_results):
        """Test that default concurrency (1) is used when not specified."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "__init__", return_value=None) as mock_init:
            with patch.object(
                AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = wrap_results(sample_attack_results)

                await atomic_attack.run_async()

                mock_init.assert_called_once_with(max_concurrency=1)

    @pytest.mark.asyncio
    async def test_run_async_passes_memory_labels(self, mock_attack, sample_objectives, sample_attack_results):
        """Test that memory labels are passed to the executor."""
        memory_labels = {"test": "attack_run", "category": "attack"}

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            memory_labels=memory_labels,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            # Check that memory_labels were passed in the call
            call_kwargs = mock_exec.call_args.kwargs
            assert "memory_labels" in call_kwargs
            assert call_kwargs["memory_labels"] == memory_labels

    @pytest.mark.asyncio
    async def test_run_async_passes_objectives(self, mock_attack, sample_objectives, sample_attack_results):
        """Test that objectives are passed to the executor."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            # Check that objectives were passed
            call_kwargs = mock_exec.call_args.kwargs
            assert "objectives" in call_kwargs
            assert call_kwargs["objectives"] == sample_objectives

    @pytest.mark.asyncio
    async def test_run_async_passes_prepended_conversation(
        self, mock_attack, sample_objectives, sample_conversation, sample_attack_results
    ):
        """Test that prepended conversation is passed to the executor."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            prepended_conversations=sample_conversation,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            # Check that prepended_conversation was passed (singular for unknown attack types)
            call_kwargs = mock_exec.call_args.kwargs
            assert "prepended_conversation" in call_kwargs
            # For unknown attack types, uses first conversation or None
            expected_conversation = sample_conversation[0] if sample_conversation else None
            assert call_kwargs["prepended_conversation"] == expected_conversation

    @pytest.mark.asyncio
    async def test_run_async_passes_attack_execute_params(self, mock_attack, sample_objectives, sample_attack_results):
        """Test that attack execute parameters are passed to the executor."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            custom_param="value",
            max_retries=3,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            # Check that custom parameters were passed
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["custom_param"] == "value"
            assert call_kwargs["max_retries"] == 3

    @pytest.mark.asyncio
    async def test_run_async_merges_all_parameters(
        self, mock_attack, sample_objectives, sample_conversation, sample_attack_results
    ):
        """Test that all parameters are merged and passed correctly."""
        memory_labels = {"test": "merge"}

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            prepended_conversations=sample_conversation,
            memory_labels=memory_labels,
            batch_size=5,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            # Verify all parameters were passed
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == mock_attack
            assert call_kwargs["objectives"] == sample_objectives
            # For unknown attack types, uses prepended_conversation (singular)
            expected_conversation = sample_conversation[0] if sample_conversation else None
            assert call_kwargs["prepended_conversation"] == expected_conversation
            assert call_kwargs["memory_labels"] == memory_labels
            assert call_kwargs["batch_size"] == 5

    @pytest.mark.asyncio
    async def test_run_async_handles_execution_failure(self, mock_attack, sample_objectives):
        """Test that execution failures are properly handled and raised."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception("Execution error")

            with pytest.raises(ValueError, match="Failed to execute atomic attack"):
                await atomic_attack.run_async()

    @pytest.mark.asyncio
    async def test_run_async_passes_return_partial_on_failure_true_by_default(
        self, mock_attack, sample_objectives, sample_attack_results
    ):
        """Test that atomic attack passes return_partial_on_failure=True by default."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            call_kwargs = mock_exec.call_args.kwargs
            assert "return_partial_on_failure" in call_kwargs
            assert call_kwargs["return_partial_on_failure"] is True

    @pytest.mark.asyncio
    async def test_run_async_respects_explicit_return_partial_on_failure(
        self, mock_attack, sample_objectives, sample_attack_results
    ):
        """Test that explicit return_partial_on_failure parameter is passed through."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async(return_partial_on_failure=False)

            call_kwargs = mock_exec.call_args.kwargs
            assert "return_partial_on_failure" in call_kwargs
            assert call_kwargs["return_partial_on_failure"] is False


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackIntegration:
    """Integration Tests for AtomicAttack."""

    @pytest.mark.asyncio
    async def test_full_attack_run_execution_flow(self, mock_attack, sample_objectives, sample_conversation):
        """Test the complete attack run execution flow end-to-end."""
        memory_labels = {"test": "integration", "attack_run": "full"}

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            prepended_conversations=sample_conversation,
            memory_labels=memory_labels,
            batch_size=2,
            atomic_attack_name="Test Attack Run",
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
            mock_exec.return_value = wrap_results(mock_results)

            attack_run_result = await atomic_attack.run_async(max_concurrency=3)

            # Verify results
            assert len(attack_run_result.completed_results) == 3
            for i, result in enumerate(attack_run_result.completed_results):
                assert result.objective == f"objective{i+1}"
                assert result.outcome == AttackOutcome.SUCCESS

            # Verify the call was made with all expected parameters
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == mock_attack
            assert call_kwargs["objectives"] == sample_objectives
            # For unknown attack types, uses prepended_conversation (singular)
            expected_conversation = sample_conversation[0] if sample_conversation else None
            assert call_kwargs["prepended_conversation"] == expected_conversation
            assert call_kwargs["memory_labels"] == memory_labels
            assert call_kwargs["batch_size"] == 2

    @pytest.mark.asyncio
    async def test_atomic_attack_with_no_optional_parameters(self, mock_attack, sample_objectives):
        """Test atomic attack with only required parameters."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=sample_objectives,
            atomic_attack_name="Test Attack Run",
        )

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
            mock_exec.return_value = wrap_results(mock_results)

            attack_run_result = await atomic_attack.run_async()

            # Verify results
            assert len(attack_run_result.completed_results) == 3

            # Verify the call was made with minimal parameters
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == mock_attack
            assert call_kwargs["objectives"] == sample_objectives
            # For unknown attack types with no prepended_conversations, should be None
            assert call_kwargs["prepended_conversation"] is None
            assert call_kwargs["memory_labels"] == {}

    @pytest.mark.asyncio
    async def test_atomic_attack_with_single_objective(self, mock_attack):
        """Test atomic attack with a single objective."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=["single_objective"],
            atomic_attack_name="Test Attack Run",
        )

        mock_result = [
            AttackResult(
                conversation_id="conv-1",
                objective="single_objective",
                attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": "1"},
                outcome=AttackOutcome.SUCCESS,
                executed_turns=1,
            )
        ]

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = wrap_results(mock_result)

            attack_run_result = await atomic_attack.run_async()

            assert len(attack_run_result.completed_results) == 1
            assert attack_run_result.completed_results[0].objective == "single_objective"

    @pytest.mark.asyncio
    async def test_atomic_attack_with_many_objectives(self, mock_attack):
        """Test atomic attack with many objectives."""
        many_objectives = [f"objective_{i}" for i in range(20)]

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            objectives=many_objectives,
            atomic_attack_name="Test Attack Run",
        )

        mock_results = [
            AttackResult(
                conversation_id=f"conv-{i}",
                objective=f"objective_{i}",
                attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": str(i)},
                outcome=AttackOutcome.SUCCESS,
                executed_turns=1,
            )
            for i in range(20)
        ]

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = wrap_results(mock_results)

            attack_run_result = await atomic_attack.run_async()

            assert len(attack_run_result.completed_results) == 20

            # Verify objectives were passed correctly
            call_kwargs = mock_exec.call_args.kwargs
            assert len(call_kwargs["objectives"]) == 20
