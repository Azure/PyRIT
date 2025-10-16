# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the scenarios.AttackRun class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import AttackExecutor, AttackStrategy
from pyrit.models import AttackOutcome, AttackResult, PromptRequestResponse
from pyrit.scenarios import AttackRun


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


@pytest.fixture
def sample_conversation():
    """Create sample conversation for testing."""
    return [
        MagicMock(spec=PromptRequestResponse),
        MagicMock(spec=PromptRequestResponse),
    ]


@pytest.mark.usefixtures("patch_central_database")
class TestAttackRunInitialization:
    """Tests for AttackRun class initialization."""

    def test_init_with_valid_params(self, mock_attack, sample_objectives):
        """Test successful initialization with valid parameters."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
        )

        assert attack_run._attack == mock_attack
        assert attack_run._objectives == sample_objectives
        assert attack_run._prepended_conversation is None
        assert attack_run._memory_labels == {}
        assert attack_run._attack_execute_params == {}

    def test_init_with_memory_labels(self, mock_attack, sample_objectives):
        """Test initialization with memory labels."""
        memory_labels = {"test": "label", "category": "attack"}

        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
            memory_labels=memory_labels,
        )

        assert attack_run._memory_labels == memory_labels

    def test_init_with_prepended_conversation(self, mock_attack, sample_objectives, sample_conversation):
        """Test initialization with prepended conversation."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
            prepended_conversation=sample_conversation,
        )

        assert attack_run._prepended_conversation == sample_conversation

    def test_init_with_attack_execute_params(self, mock_attack, sample_objectives):
        """Test initialization with additional attack execute parameters."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
            max_retries=5,
            custom_param="value",
        )

        assert attack_run._attack_execute_params["max_retries"] == 5
        assert attack_run._attack_execute_params["custom_param"] == "value"

    def test_init_with_all_parameters(self, mock_attack, sample_objectives, sample_conversation):
        """Test initialization with all parameters."""
        memory_labels = {"test": "comprehensive"}

        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
            prepended_conversation=sample_conversation,
            memory_labels=memory_labels,
            batch_size=10,
            timeout=30,
        )

        assert attack_run._attack == mock_attack
        assert attack_run._objectives == sample_objectives
        assert attack_run._prepended_conversation == sample_conversation
        assert attack_run._memory_labels == memory_labels
        assert attack_run._attack_execute_params["batch_size"] == 10
        assert attack_run._attack_execute_params["timeout"] == 30

    def test_init_fails_with_empty_objectives(self, mock_attack):
        """Test that initialization fails when objectives list is empty."""
        with pytest.raises(ValueError, match="objectives list cannot be empty"):
            AttackRun(
                attack=mock_attack,
                objectives=[],
            )


@pytest.mark.usefixtures("patch_central_database")
class TestAttackRunExecution:
    """Tests for AttackRun execution methods."""

    @pytest.mark.asyncio
    async def test_run_async_with_valid_attack_run(self, mock_attack, sample_objectives, sample_attack_results):
        """Test successful execution of an attack run."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
        )

        # Mock the executor
        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            results = await attack_run.run_async()

            assert len(results) == 3
            assert results == sample_attack_results
            mock_exec.assert_called_once()

            # Verify the attack was passed correctly
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == mock_attack

    @pytest.mark.asyncio
    async def test_run_async_with_custom_concurrency(self, mock_attack, sample_objectives, sample_attack_results):
        """Test execution with custom max_concurrency."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
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
    async def test_run_async_with_default_concurrency(self, mock_attack, sample_objectives, sample_attack_results):
        """Test that default concurrency (1) is used when not specified."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
        )

        with patch.object(AttackExecutor, "__init__", return_value=None) as mock_init:
            with patch.object(
                AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = sample_attack_results

                await attack_run.run_async()

                mock_init.assert_called_once_with(max_concurrency=1)

    @pytest.mark.asyncio
    async def test_run_async_passes_memory_labels(self, mock_attack, sample_objectives, sample_attack_results):
        """Test that memory labels are passed to the executor."""
        memory_labels = {"test": "attack_run", "category": "attack"}

        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
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
    async def test_run_async_passes_objectives(self, mock_attack, sample_objectives, sample_attack_results):
        """Test that objectives are passed to the executor."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            await attack_run.run_async()

            # Check that objectives were passed
            call_kwargs = mock_exec.call_args.kwargs
            assert "objectives" in call_kwargs
            assert call_kwargs["objectives"] == sample_objectives

    @pytest.mark.asyncio
    async def test_run_async_passes_prepended_conversation(
        self, mock_attack, sample_objectives, sample_conversation, sample_attack_results
    ):
        """Test that prepended conversation is passed to the executor."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
            prepended_conversation=sample_conversation,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            await attack_run.run_async()

            # Check that prepended_conversation was passed
            call_kwargs = mock_exec.call_args.kwargs
            assert "prepended_conversation" in call_kwargs
            assert call_kwargs["prepended_conversation"] == sample_conversation

    @pytest.mark.asyncio
    async def test_run_async_passes_attack_execute_params(self, mock_attack, sample_objectives, sample_attack_results):
        """Test that attack execute parameters are passed to the executor."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
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
    async def test_run_async_merges_all_parameters(
        self, mock_attack, sample_objectives, sample_conversation, sample_attack_results
    ):
        """Test that all parameters are merged and passed correctly."""
        memory_labels = {"test": "merge"}

        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
            prepended_conversation=sample_conversation,
            memory_labels=memory_labels,
            batch_size=5,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = sample_attack_results

            await attack_run.run_async()

            # Verify all parameters were passed
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == mock_attack
            assert call_kwargs["objectives"] == sample_objectives
            assert call_kwargs["prepended_conversation"] == sample_conversation
            assert call_kwargs["memory_labels"] == memory_labels
            assert call_kwargs["batch_size"] == 5

    @pytest.mark.asyncio
    async def test_run_async_handles_execution_failure(self, mock_attack, sample_objectives):
        """Test that execution failures are properly handled and raised."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
        )

        with patch.object(AttackExecutor, "execute_multi_objective_attack_async", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception("Execution error")

            with pytest.raises(ValueError, match="Failed to execute attack run"):
                await attack_run.run_async()


@pytest.mark.usefixtures("patch_central_database")
class TestAttackRunIntegration:
    """Integration tests for AttackRun."""

    @pytest.mark.asyncio
    async def test_full_attack_run_execution_flow(self, mock_attack, sample_objectives, sample_conversation):
        """Test the complete attack run execution flow end-to-end."""
        memory_labels = {"test": "integration", "attack_run": "full"}

        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
            prepended_conversation=sample_conversation,
            memory_labels=memory_labels,
            batch_size=2,
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
            assert call_kwargs["attack"] == mock_attack
            assert call_kwargs["objectives"] == sample_objectives
            assert call_kwargs["prepended_conversation"] == sample_conversation
            assert call_kwargs["memory_labels"] == memory_labels
            assert call_kwargs["batch_size"] == 2

    @pytest.mark.asyncio
    async def test_attack_run_with_no_optional_parameters(self, mock_attack, sample_objectives):
        """Test attack run with only required parameters."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=sample_objectives,
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
            mock_exec.return_value = mock_results

            results = await attack_run.run_async()

            # Verify results
            assert len(results) == 3

            # Verify the call was made with minimal parameters
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == mock_attack
            assert call_kwargs["objectives"] == sample_objectives
            assert call_kwargs["prepended_conversation"] is None
            assert call_kwargs["memory_labels"] == {}

    @pytest.mark.asyncio
    async def test_attack_run_with_single_objective(self, mock_attack):
        """Test attack run with a single objective."""
        attack_run = AttackRun(
            attack=mock_attack,
            objectives=["single_objective"],
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
            mock_exec.return_value = mock_result

            results = await attack_run.run_async()

            assert len(results) == 1
            assert results[0].objective == "single_objective"

    @pytest.mark.asyncio
    async def test_attack_run_with_many_objectives(self, mock_attack):
        """Test attack run with many objectives."""
        many_objectives = [f"objective_{i}" for i in range(20)]

        attack_run = AttackRun(
            attack=mock_attack,
            objectives=many_objectives,
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
            mock_exec.return_value = mock_results

            results = await attack_run.run_async()

            assert len(results) == 20

            # Verify objectives were passed correctly
            call_kwargs = mock_exec.call_args.kwargs
            assert len(call_kwargs["objectives"]) == 20
