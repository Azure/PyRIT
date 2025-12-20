# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the scenarios.AtomicAttack class."""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import AttackExecutor, AttackStrategy
from pyrit.executor.attack.core import AttackExecutorResult
from pyrit.models import AttackOutcome, AttackResult, SeedGroup, SeedObjective, SeedPrompt
from pyrit.scenario import AtomicAttack


@pytest.fixture
def mock_attack():
    """Create a mock AttackStrategy for testing."""
    return MagicMock(spec=AttackStrategy)


@pytest.fixture
def sample_seed_groups():
    """Create sample seed groups with objectives for testing."""
    return [
        SeedGroup(
            seeds=[
                SeedObjective(value="objective1"),
                SeedPrompt(value="prompt1"),
            ]
        ),
        SeedGroup(
            seeds=[
                SeedObjective(value="objective2"),
                SeedPrompt(value="prompt2"),
            ]
        ),
        SeedGroup(
            seeds=[
                SeedObjective(value="objective3"),
                SeedPrompt(value="prompt3"),
            ]
        ),
    ]


@pytest.fixture
def sample_seed_groups_without_objectives():
    """Create sample seed groups without objectives for testing."""
    return [
        SeedGroup(
            seeds=[
                SeedPrompt(value="prompt1"),
            ]
        ),
    ]


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


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackInitialization:
    """Tests for AtomicAttack class initialization."""

    def test_init_with_valid_params(self, mock_attack, sample_seed_groups):
        """Test successful initialization with valid parameters."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        assert atomic_attack._attack == mock_attack
        assert atomic_attack._seed_groups == sample_seed_groups
        assert atomic_attack._memory_labels == {}
        assert atomic_attack._attack_execute_params == {}

    def test_init_with_memory_labels(self, mock_attack, sample_seed_groups):
        """Test initialization with memory labels."""
        memory_labels = {"test": "label", "category": "attack"}

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            memory_labels=memory_labels,
            atomic_attack_name="Test Attack Run",
        )

        assert atomic_attack._memory_labels == memory_labels

    def test_init_with_attack_execute_params(self, mock_attack, sample_seed_groups):
        """Test initialization with additional attack execute parameters."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            max_retries=5,
            custom_param="value",
            atomic_attack_name="Test Attack Run",
        )

        assert atomic_attack._attack_execute_params["max_retries"] == 5
        assert atomic_attack._attack_execute_params["custom_param"] == "value"

    def test_init_with_all_parameters(self, mock_attack, sample_seed_groups):
        """Test initialization with all parameters."""
        memory_labels = {"test": "comprehensive"}

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            memory_labels=memory_labels,
            batch_size=10,
            timeout=30,
            atomic_attack_name="Test Attack Run",
        )

        assert atomic_attack._attack == mock_attack
        assert atomic_attack._seed_groups == sample_seed_groups
        assert atomic_attack._memory_labels == memory_labels
        assert atomic_attack._attack_execute_params["batch_size"] == 10
        assert atomic_attack._attack_execute_params["timeout"] == 30

    def test_init_fails_with_empty_seed_groups(self, mock_attack):
        """Test that initialization fails when seed_groups list is empty."""
        with pytest.raises(ValueError, match="seed_groups list cannot be empty"):
            AtomicAttack(
                attack=mock_attack,
                seed_groups=[],
                atomic_attack_name="Test Attack Run",
            )

    def test_init_fails_with_seed_group_missing_objective(
        self, mock_attack, sample_seed_groups_without_objectives
    ):
        """Test that initialization fails when a seed group is missing an objective."""
        with pytest.raises(ValueError, match="SeedGroup at index 0 is missing an objective"):
            AtomicAttack(
                attack=mock_attack,
                seed_groups=sample_seed_groups_without_objectives,
                atomic_attack_name="Test Attack Run",
            )

    def test_objectives_property_returns_values_from_seed_groups(self, mock_attack, sample_seed_groups):
        """Test that the objectives property returns values from seed groups."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        assert atomic_attack.objectives == ["objective1", "objective2", "objective3"]

    def test_seed_groups_property_returns_copy(self, mock_attack, sample_seed_groups):
        """Test that the seed_groups property returns a copy."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        returned_groups = atomic_attack.seed_groups
        assert returned_groups == sample_seed_groups
        assert returned_groups is not atomic_attack._seed_groups


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackExecution:
    """Tests for AtomicAttack execution methods."""

    @pytest.mark.asyncio
    async def test_run_async_with_valid_atomic_attack(
        self, mock_attack, sample_seed_groups, sample_attack_results
    ):
        """Test successful execution of an atomic attack."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
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
    async def test_run_async_with_custom_concurrency(
        self, mock_attack, sample_seed_groups, sample_attack_results
    ):
        """Test execution with custom max_concurrency for atomic attack."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "__init__", return_value=None) as mock_init:
            with patch.object(
                AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = wrap_results(sample_attack_results)

                result = await atomic_attack.run_async(max_concurrency=5)

                mock_init.assert_called_once_with(max_concurrency=5)
                assert len(result.completed_results) == 3

    @pytest.mark.asyncio
    async def test_run_async_with_default_concurrency(
        self, mock_attack, sample_seed_groups, sample_attack_results
    ):
        """Test that default concurrency (1) is used when not specified."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(AttackExecutor, "__init__", return_value=None) as mock_init:
            with patch.object(
                AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
            ) as mock_exec:
                mock_exec.return_value = wrap_results(sample_attack_results)

                await atomic_attack.run_async()

                mock_init.assert_called_once_with(max_concurrency=1)

    @pytest.mark.asyncio
    async def test_run_async_passes_memory_labels(
        self, mock_attack, sample_seed_groups, sample_attack_results
    ):
        """Test that memory labels are passed to the executor."""
        memory_labels = {"test": "attack_run", "category": "attack"}

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            memory_labels=memory_labels,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            call_kwargs = mock_exec.call_args.kwargs
            assert "memory_labels" in call_kwargs
            assert call_kwargs["memory_labels"] == memory_labels

    @pytest.mark.asyncio
    async def test_run_async_passes_seed_groups(
        self, mock_attack, sample_seed_groups, sample_attack_results
    ):
        """Test that seed_groups are passed to the executor."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            call_kwargs = mock_exec.call_args.kwargs
            assert "seed_groups" in call_kwargs
            assert call_kwargs["seed_groups"] == sample_seed_groups

    @pytest.mark.asyncio
    async def test_run_async_passes_attack_execute_params(
        self, mock_attack, sample_seed_groups, sample_attack_results
    ):
        """Test that attack execute parameters are passed to the executor."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            custom_param="value",
            max_retries=3,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["custom_param"] == "value"
            assert call_kwargs["max_retries"] == 3

    @pytest.mark.asyncio
    async def test_run_async_merges_all_parameters(
        self, mock_attack, sample_seed_groups, sample_attack_results
    ):
        """Test that all parameters are merged and passed correctly."""
        memory_labels = {"test": "merge"}

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            memory_labels=memory_labels,
            batch_size=5,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == mock_attack
            assert call_kwargs["seed_groups"] == sample_seed_groups
            assert call_kwargs["memory_labels"] == memory_labels
            assert call_kwargs["batch_size"] == 5

    @pytest.mark.asyncio
    async def test_run_async_handles_execution_failure(self, mock_attack, sample_seed_groups):
        """Test that execution failures are properly handled and raised."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.side_effect = Exception("Execution error")

            with pytest.raises(ValueError, match="Failed to execute atomic attack"):
                await atomic_attack.run_async()

    @pytest.mark.asyncio
    async def test_run_async_passes_return_partial_on_failure_true_by_default(
        self, mock_attack, sample_seed_groups, sample_attack_results
    ):
        """Test that atomic attack passes return_partial_on_failure=True by default."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            call_kwargs = mock_exec.call_args.kwargs
            assert "return_partial_on_failure" in call_kwargs
            assert call_kwargs["return_partial_on_failure"] is True

    @pytest.mark.asyncio
    async def test_run_async_respects_explicit_return_partial_on_failure(
        self, mock_attack, sample_seed_groups, sample_attack_results
    ):
        """Test that explicit return_partial_on_failure parameter is passed through."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async(return_partial_on_failure=False)

            call_kwargs = mock_exec.call_args.kwargs
            assert "return_partial_on_failure" in call_kwargs
            assert call_kwargs["return_partial_on_failure"] is False


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackIntegration:
    """Integration Tests for AtomicAttack."""

    @pytest.mark.asyncio
    async def test_full_attack_run_execution_flow(self, mock_attack, sample_seed_groups):
        """Test the complete attack run execution flow end-to-end."""
        memory_labels = {"test": "integration", "attack_run": "full"}

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            memory_labels=memory_labels,
            batch_size=2,
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

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(mock_results)

            attack_run_result = await atomic_attack.run_async(max_concurrency=3)

            assert len(attack_run_result.completed_results) == 3
            for i, result in enumerate(attack_run_result.completed_results):
                assert result.objective == f"objective{i+1}"
                assert result.outcome == AttackOutcome.SUCCESS

            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["attack"] == mock_attack
            assert call_kwargs["seed_groups"] == sample_seed_groups
            assert call_kwargs["memory_labels"] == memory_labels
            assert call_kwargs["batch_size"] == 2

    @pytest.mark.asyncio
    async def test_atomic_attack_with_single_seed_group(self, mock_attack):
        """Test atomic attack with a single seed group."""
        single_seed_group = [
            SeedGroup(
                seeds=[
                    SeedObjective(value="single_objective"),
                    SeedPrompt(value="single_prompt"),
                ]
            )
        ]

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=single_seed_group,
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

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(mock_result)

            attack_run_result = await atomic_attack.run_async()

            assert len(attack_run_result.completed_results) == 1
            assert attack_run_result.completed_results[0].objective == "single_objective"

    @pytest.mark.asyncio
    async def test_atomic_attack_with_many_seed_groups(self, mock_attack):
        """Test atomic attack with many seed groups."""
        many_seed_groups = [
            SeedGroup(
                seeds=[
                    SeedObjective(value=f"objective_{i}"),
                    SeedPrompt(value=f"prompt_{i}"),
                ]
            )
            for i in range(20)
        ]

        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=many_seed_groups,
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

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(mock_results)

            attack_run_result = await atomic_attack.run_async()

            assert len(attack_run_result.completed_results) == 20

            call_kwargs = mock_exec.call_args.kwargs
            assert len(call_kwargs["seed_groups"]) == 20


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackExecutorParamCompatibility:
    """Tests to verify AtomicAttack passes parameters compatible with AttackExecutor."""

    def test_atomic_attack_passes_expected_executor_params(self, mock_attack, sample_seed_groups):
        """
        Test that AtomicAttack.run_async passes all expected parameters
        to execute_attack_from_seed_groups_async.
        """
        # Get the signature of execute_attack_from_seed_groups_async
        executor_method = AttackExecutor.execute_attack_from_seed_groups_async
        sig = inspect.signature(executor_method)

        # These are the parameters that execute_attack_from_seed_groups_async accepts
        expected_params = set(sig.parameters.keys()) - {"self"}

        # Verify the parameters we know AtomicAttack should pass
        required_from_atomic_attack = {
            "attack",
            "seed_groups",
            "memory_labels",
            "return_partial_on_failure",
        }

        # All required params should be in the executor method signature
        assert required_from_atomic_attack.issubset(
            expected_params
        ), f"Missing expected params in executor: {required_from_atomic_attack - expected_params}"

    @pytest.mark.asyncio
    async def test_run_async_only_passes_valid_executor_params(
        self, mock_attack, sample_seed_groups, sample_attack_results
    ):
        """
        Test that run_async doesn't pass parameters that the executor doesn't accept.
        The executor has strict_param_matching so invalid params would cause failures.
        """
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=sample_seed_groups,
            atomic_attack_name="Test Attack Run",
        )

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(sample_attack_results)

            await atomic_attack.run_async()

            call_kwargs = mock_exec.call_args.kwargs

            # Verify essential params are present
            assert "attack" in call_kwargs
            assert "seed_groups" in call_kwargs
            assert "memory_labels" in call_kwargs
            assert "return_partial_on_failure" in call_kwargs


@pytest.mark.usefixtures("patch_central_database")
class TestAtomicAttackWithMessages:
    """Tests for AtomicAttack with seed groups containing multi-turn messages."""

    @pytest.fixture
    def seed_groups_with_messages(self):
        """Create seed groups with multi-turn message sequences for testing."""
        return [
            SeedGroup(
                seeds=[
                    SeedObjective(value="multi_turn_objective_1"),
                    SeedPrompt(value="First message", data_type="text", sequence=0, role="user"),
                    SeedPrompt(value="Second message", data_type="text", sequence=1, role="user"),
                    SeedPrompt(value="Third message", data_type="text", sequence=2, role="user"),
                ]
            ),
            SeedGroup(
                seeds=[
                    SeedObjective(value="multi_turn_objective_2"),
                    SeedPrompt(value="Message A", data_type="text", sequence=0, role="user"),
                    SeedPrompt(value="Message B", data_type="text", sequence=1, role="user"),
                ]
            ),
        ]

    @pytest.fixture
    def mixed_seed_groups(self):
        """Create seed groups where some have messages and some don't."""
        return [
            # No messages (just objective)
            SeedGroup(seeds=[SeedObjective(value="simple_objective")]),
            # With messages - roles required for multi-sequence
            SeedGroup(
                seeds=[
                    SeedObjective(value="objective_with_messages"),
                    SeedPrompt(value="Message 1", data_type="text", sequence=0, role="user"),
                    SeedPrompt(value="Message 2", data_type="text", sequence=1, role="user"),
                ]
            ),
        ]

    def test_init_with_seed_groups_with_messages(self, mock_attack, seed_groups_with_messages):
        """Test that AtomicAttack initializes correctly with seed groups containing messages."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=seed_groups_with_messages,
            atomic_attack_name="Multi-turn Attack",
        )

        assert len(atomic_attack.seed_groups) == 2
        assert atomic_attack.objectives == ["multi_turn_objective_1", "multi_turn_objective_2"]

        # Verify seed groups have user messages
        for sg in atomic_attack.seed_groups:
            assert len(sg.user_messages) > 0

    def test_seed_groups_user_messages_property(self, mock_attack, seed_groups_with_messages):
        """Test that seed group user_messages are accessible and have correct content."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=seed_groups_with_messages,
            atomic_attack_name="Multi-turn Attack",
        )

        sg1 = atomic_attack.seed_groups[0]
        sg2 = atomic_attack.seed_groups[1]

        # First seed group has 3 user messages
        assert len(sg1.user_messages) == 3
        assert sg1.user_messages[0].message_pieces[0].original_value == "First message"
        assert sg1.user_messages[1].message_pieces[0].original_value == "Second message"
        assert sg1.user_messages[2].message_pieces[0].original_value == "Third message"

        # Second seed group has 2 user messages
        assert len(sg2.user_messages) == 2
        assert sg2.user_messages[0].message_pieces[0].original_value == "Message A"
        assert sg2.user_messages[1].message_pieces[0].original_value == "Message B"

    @pytest.mark.asyncio
    async def test_run_async_passes_seed_groups_with_messages(
        self, mock_attack, seed_groups_with_messages
    ):
        """Test that run_async correctly passes seed groups with messages to executor."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=seed_groups_with_messages,
            atomic_attack_name="Multi-turn Attack",
        )

        mock_results = [
            AttackResult(
                conversation_id=f"conv-{i}",
                objective=seed_groups_with_messages[i].objective.value,
                attack_identifier={"__type__": "TestAttack", "__module__": "test", "id": str(i)},
                outcome=AttackOutcome.SUCCESS,
                executed_turns=len(seed_groups_with_messages[i].user_messages),
            )
            for i in range(2)
        ]

        with patch.object(
            AttackExecutor, "execute_attack_from_seed_groups_async", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = wrap_results(mock_results)

            result = await atomic_attack.run_async()

            assert len(result.completed_results) == 2

            # Verify seed groups were passed correctly
            call_kwargs = mock_exec.call_args.kwargs
            passed_seed_groups = call_kwargs["seed_groups"]
            assert len(passed_seed_groups) == 2

            # Verify user messages are preserved in passed seed groups
            assert len(passed_seed_groups[0].user_messages) == 3
            assert len(passed_seed_groups[1].user_messages) == 2

    def test_init_with_mixed_seed_groups(self, mock_attack, mixed_seed_groups):
        """Test that AtomicAttack handles mixed seed groups (some with user_messages, some without)."""
        atomic_attack = AtomicAttack(
            attack=mock_attack,
            seed_groups=mixed_seed_groups,
            atomic_attack_name="Mixed Attack",
        )

        assert len(atomic_attack.seed_groups) == 2

        # First has no user_messages (empty list)
        assert len(atomic_attack.seed_groups[0].user_messages) == 0

        # Second has user_messages
        assert len(atomic_attack.seed_groups[1].user_messages) == 2
