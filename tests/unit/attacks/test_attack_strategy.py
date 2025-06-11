# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.attacks.base.attack_result import AttackOutcome
from pyrit.attacks.base.attack_strategy import AttackStrategy, AttackStrategyLogAdapter
from pyrit.exceptions.exception_classes import (
    AttackExecutionException,
    AttackValidationException,
)


@pytest.fixture
def basic_context():
    """Create a basic mock context for testing"""
    context = MagicMock()
    context.objective = "Test objective"
    context.memory_labels = {"test": "label"}
    return context


@pytest.fixture
def mock_default_values():
    """Mock default values for memory labels"""
    with patch("pyrit.attacks.base.attack_strategy.default_values") as mock:
        mock.get_non_required_value.return_value = '{"test_label": "test_value"}'
        yield mock


@pytest.fixture
def mock_attack_strategy(mock_default_values):
    """Create a mock attack strategy with all abstract methods mocked"""

    # Create a concrete subclass with mocked abstract methods
    class TestableAttackStrategy(AttackStrategy):
        def __init__(self, **kwargs):
            # Use the root logger to ensure caplog can capture the logs
            super().__init__(logger=logging.getLogger(), **kwargs)

        # Mock all abstract methods
        _validate_context = MagicMock()
        _setup_async = AsyncMock()
        _perform_attack_async = AsyncMock()
        _teardown_async = AsyncMock()

    strategy = TestableAttackStrategy()

    # Configure default return value for _perform_attack_async with required attributes
    mock_result = MagicMock()
    mock_result.outcome = AttackOutcome.SUCCESS
    mock_result.outcome_reason = "Test successful"
    mock_result.execution_time_ms = None  # Will be set by execute_async

    # Cast to AsyncMock to satisfy type checker
    from typing import cast

    cast(AsyncMock, strategy._perform_attack_async).return_value = mock_result

    return strategy


@pytest.mark.usefixtures("patch_central_database")
class TestAttackStrategyInitialization:
    """Tests for AttackStrategy initialization"""

    def test_init_with_default_parameters(self, mock_default_values):
        class ConcreteStrategy(AttackStrategy):
            _validate_context = MagicMock()
            _setup_async = AsyncMock()
            _perform_attack_async = AsyncMock()
            _teardown_async = AsyncMock()

        strategy = ConcreteStrategy()

        assert strategy._id is not None
        assert isinstance(strategy._id, uuid.UUID)
        assert strategy._memory is not None
        assert strategy._memory_labels == {"test_label": "test_value"}
        assert isinstance(strategy._logger, AttackStrategyLogAdapter)

    def test_init_with_custom_logger(self, mock_default_values):
        class ConcreteStrategy(AttackStrategy):
            _validate_context = MagicMock()
            _setup_async = AsyncMock()
            _perform_attack_async = AsyncMock()
            _teardown_async = AsyncMock()

        custom_logger = logging.getLogger("custom_test_logger")
        strategy = ConcreteStrategy(logger=custom_logger)

        assert strategy._logger.logger == custom_logger

    @pytest.mark.parametrize(
        "memory_labels_str,expected",
        [
            ('{"env1": "value1"}', {"env1": "value1"}),
            ('{"env1": "value1", "env2": "value2"}', {"env1": "value1", "env2": "value2"}),
            ("{}", {}),
            (None, {}),
        ],
    )
    def test_init_with_various_memory_labels(self, memory_labels_str, expected):
        with patch("pyrit.attacks.base.attack_strategy.default_values") as mock:
            mock.get_non_required_value.return_value = memory_labels_str

            class ConcreteStrategy(AttackStrategy):
                _validate_context = MagicMock()
                _setup_async = AsyncMock()
                _perform_attack_async = AsyncMock()
                _teardown_async = AsyncMock()

            strategy = ConcreteStrategy()
            assert strategy._memory_labels == expected

    def test_get_identifier_format(self, mock_attack_strategy):
        identifier = mock_attack_strategy.get_identifier()

        assert "__type__" in identifier
        assert "__module__" in identifier
        assert "id" in identifier
        assert identifier["__type__"] == "TestableAttackStrategy"
        assert identifier["id"] == str(mock_attack_strategy._id)


@pytest.mark.usefixtures("patch_central_database")
class TestAttackExecution:
    """Tests for the main attack execution logic (execute_async)"""

    @pytest.mark.asyncio
    async def test_execute_async_successful_flow(self, mock_attack_strategy, basic_context):
        result = await mock_attack_strategy.execute_async(context=basic_context)

        # Verify all lifecycle methods were called
        mock_attack_strategy._validate_context.assert_called_once_with(context=basic_context)
        mock_attack_strategy._setup_async.assert_called_once_with(context=basic_context)
        mock_attack_strategy._perform_attack_async.assert_called_once_with(context=basic_context)
        mock_attack_strategy._teardown_async.assert_called_once_with(context=basic_context)

        # Verify result
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.outcome_reason == "Test successful"
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_async_validation_failure(self, mock_attack_strategy, basic_context):
        mock_attack_strategy._validate_context.side_effect = ValueError("Validation failed")

        with pytest.raises(AttackValidationException) as exc_info:
            await mock_attack_strategy.execute_async(context=basic_context)

        # Verify error details
        assert "Context validation failed" in str(exc_info.value)
        assert exc_info.value.context_info["attack_type"] == "TestableAttackStrategy"
        assert exc_info.value.context_info["original_error"] == "Validation failed"

        # Verify lifecycle - only validation should be called
        mock_attack_strategy._validate_context.assert_called_once()
        mock_attack_strategy._setup_async.assert_not_called()
        mock_attack_strategy._perform_attack_async.assert_not_called()
        mock_attack_strategy._teardown_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_async_setup_failure_calls_teardown(self, mock_attack_strategy, basic_context):
        mock_attack_strategy._setup_async.side_effect = RuntimeError("Setup failed")

        with pytest.raises(AttackExecutionException) as exc_info:
            await mock_attack_strategy.execute_async(context=basic_context)

        # Verify error details
        assert "Unexpected error during attack execution" in str(exc_info.value)
        assert exc_info.value.attack_name == "TestableAttackStrategy"
        assert exc_info.value.objective == "Test objective"

        # Verify lifecycle - teardown should still be called
        mock_attack_strategy._validate_context.assert_called_once()
        mock_attack_strategy._setup_async.assert_called_once()
        mock_attack_strategy._perform_attack_async.assert_not_called()
        mock_attack_strategy._teardown_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_async_perform_failure_calls_teardown(self, mock_attack_strategy, basic_context):
        mock_attack_strategy._perform_attack_async.side_effect = RuntimeError("Attack failed")

        with pytest.raises(AttackExecutionException):
            await mock_attack_strategy.execute_async(context=basic_context)

        # Verify lifecycle - teardown should still be called
        mock_attack_strategy._validate_context.assert_called_once()
        mock_attack_strategy._setup_async.assert_called_once()
        mock_attack_strategy._perform_attack_async.assert_called_once()
        mock_attack_strategy._teardown_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_async_teardown_failure_propagates(self, mock_attack_strategy, basic_context):
        mock_attack_strategy._teardown_async.side_effect = RuntimeError("Teardown failed")

        # Teardown failures should still propagate but after being called
        with pytest.raises(AttackExecutionException):
            await mock_attack_strategy.execute_async(context=basic_context)

        # All methods should have been called
        mock_attack_strategy._validate_context.assert_called_once()
        mock_attack_strategy._setup_async.assert_called_once()
        mock_attack_strategy._perform_attack_async.assert_called_once()
        mock_attack_strategy._teardown_async.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "existing_exception",
        [
            AttackValidationException(message="Existing validation error"),
            AttackExecutionException(message="Existing execution error"),
        ],
    )
    async def test_execute_async_preserves_specific_exceptions(
        self, mock_attack_strategy, basic_context, existing_exception
    ):
        mock_attack_strategy._perform_attack_async.side_effect = existing_exception

        with pytest.raises(type(existing_exception)) as exc_info:
            await mock_attack_strategy.execute_async(context=basic_context)

        # Should preserve the original exception type
        assert exc_info.value is existing_exception

    @pytest.mark.asyncio
    async def test_execute_async_sets_execution_time(self, mock_attack_strategy, basic_context):
        """Test that execute_async properly sets execution_time_ms on the result"""

        # Mock the perform_attack_async to take some time
        async def delayed_attack(context):
            import asyncio

            await asyncio.sleep(0.01)  # Sleep for 10ms to ensure measurable time
            mock_result = MagicMock()
            mock_result.outcome = AttackOutcome.SUCCESS
            mock_result.outcome_reason = "Test successful"
            mock_result.execution_time_ms = None  # Should be set by execute_async
            return mock_result

        mock_attack_strategy._perform_attack_async = delayed_attack

        # Execute the attack
        result = await mock_attack_strategy.execute_async(context=basic_context)

        # Verify execution time is set and reasonable
        assert result.execution_time_ms is not None
        assert isinstance(result.execution_time_ms, int)
        assert result.execution_time_ms >= 0  # Should be non-negative
        assert result.execution_time_ms < 1000  # Should be less than 1 second for this test

        # Verify it overwrites any existing value
        assert result.execution_time_ms is not None  # Was None before execute_async set it

    @pytest.mark.asyncio
    async def test_execute_async_overwrites_existing_execution_time(self, mock_attack_strategy, basic_context):
        """Test that execute_async overwrites any pre-existing execution_time_ms value"""
        # Configure mock result with a pre-existing execution time
        mock_result = MagicMock()
        mock_result.outcome = AttackOutcome.SUCCESS
        mock_result.outcome_reason = "Test successful"
        mock_result.execution_time_ms = 999999  # Pre-existing value that should be overwritten

        from typing import cast

        cast(AsyncMock, mock_attack_strategy._perform_attack_async).return_value = mock_result

        # Execute the attack
        result = await mock_attack_strategy.execute_async(context=basic_context)

        # Verify execution time was overwritten
        assert result.execution_time_ms is not None
        assert result.execution_time_ms != 999999  # Should have been overwritten
        assert result.execution_time_ms >= 0  # Should be a valid time measurement


@pytest.mark.usefixtures("patch_central_database")
class TestLogging:
    """Tests for logging functionality"""

    @pytest.mark.asyncio
    async def test_logging_during_lifecycle(self, mock_attack_strategy, basic_context, caplog):
        # Ensure we're capturing at the root logger level
        with caplog.at_level(logging.DEBUG, logger=""):
            await mock_attack_strategy.execute_async(context=basic_context)

        # Check for expected log messages
        log_messages = [record.message for record in caplog.records]

        # Should log validation, setup, execution, and teardown
        assert any("Validating context" in msg for msg in log_messages)
        assert any("Setting up attack strategy" in msg for msg in log_messages)
        assert any("Performing attack: TestableAttackStrategy" in msg for msg in log_messages)
        assert any("Tearing down attack strategy" in msg for msg in log_messages)
        assert any("Attack execution completed" in msg for msg in log_messages)
        assert any("achieved the objective" in msg for msg in log_messages)  # From _log_attack_outcome

    def test_logger_adapter_adds_context(self, mock_attack_strategy, caplog):
        # Ensure we're capturing at the root logger level
        with caplog.at_level(logging.DEBUG, logger=""):
            mock_attack_strategy._logger.debug("Test log message")

        # Logger adapter should add strategy info to logs
        assert len(caplog.records) == 1
        log_message = caplog.records[0].getMessage()
        assert "[TestableAttackStrategy" in log_message
        assert "Test log message" in log_message

    def test_logger_adapter_process_with_context(self):
        base_logger = logging.getLogger("test")
        # Initialize adapter with extra context containing strategy info
        adapter = AttackStrategyLogAdapter(base_logger, {"strategy_name": "TestStrategy", "strategy_id": "12345"})

        msg, kwargs = adapter.process("Test message", {})

        assert msg == "[TestStrategy (ID: 12345)] Test message"
        assert kwargs == {}

    def test_logger_adapter_process_without_context(self):
        base_logger = logging.getLogger("test")
        adapter = AttackStrategyLogAdapter(base_logger, {})

        msg, kwargs = adapter.process("Test message", {})

        assert msg == "Test message"
        assert kwargs == {}


@pytest.mark.usefixtures("patch_central_database")
class TestConcurrency:
    """Tests for concurrent execution scenarios"""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_executions(self, mock_default_values):
        # Create multiple independent strategy instances
        strategies = []
        for _ in range(5):

            class ConcreteStrategy(AttackStrategy):
                _validate_context = MagicMock()
                _setup_async = AsyncMock()
                _perform_attack_async = AsyncMock()
                _teardown_async = AsyncMock()

            strategy = ConcreteStrategy()
            # Configure mock result with required attributes
            mock_result = MagicMock()
            mock_result.outcome = AttackOutcome.SUCCESS
            mock_result.outcome_reason = "Test successful"
            mock_result.execution_time_ms = None

            # Cast to AsyncMock to satisfy type checker
            from typing import cast

            cast(AsyncMock, strategy._perform_attack_async).return_value = mock_result
            strategies.append(strategy)

        # Create multiple mock contexts
        contexts = [MagicMock(objective=f"Objective {i}") for i in range(5)]

        # Execute concurrently
        tasks = [strategy.execute_async(context=ctx) for strategy, ctx in zip(strategies, contexts)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 5
        assert all(r.outcome == AttackOutcome.SUCCESS for r in results)
        assert all(r.execution_time_ms >= 0 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_execution_isolation(self, mock_attack_strategy):
        """Ensure concurrent executions don't interfere with each other"""
        # Create contexts with different objectives
        context1 = MagicMock(objective="Objective 1")
        context2 = MagicMock(objective="Objective 2")

        # Configure return values with required attributes
        result1 = MagicMock(id="result1")
        result1.outcome = AttackOutcome.SUCCESS
        result1.outcome_reason = "Result 1 successful"
        result1.execution_time_ms = None

        result2 = MagicMock(id="result2")
        result2.outcome = AttackOutcome.FAILURE
        result2.outcome_reason = "Result 2 failed"
        result2.execution_time_ms = None

        # Cast to AsyncMock to satisfy type checker
        from typing import cast

        cast(AsyncMock, mock_attack_strategy._perform_attack_async).side_effect = [result1, result2]

        # Execute concurrently
        results = await asyncio.gather(
            mock_attack_strategy.execute_async(context=context1), mock_attack_strategy.execute_async(context=context2)
        )

        # Both should succeed with correct results
        assert len(results) == 2
        assert results[0].id == "result1"
        assert results[0].outcome == AttackOutcome.SUCCESS
        assert results[0].execution_time_ms >= 0
        assert results[1].id == "result2"
        assert results[1].outcome == AttackOutcome.FAILURE
        assert results[1].execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_attack_outcome_logging(self, mock_attack_strategy, basic_context, caplog):
        """Test that different attack outcomes are logged correctly"""
        # Test SUCCESS outcome
        with caplog.at_level(logging.INFO, logger=""):
            await mock_attack_strategy.execute_async(context=basic_context)
        assert any("achieved the objective" in record.message for record in caplog.records)

        # Test FAILURE outcome
        caplog.clear()
        mock_result = MagicMock()
        mock_result.outcome = AttackOutcome.FAILURE
        mock_result.outcome_reason = "Target refused"
        mock_result.execution_time_ms = None

        # Cast to AsyncMock to satisfy type checker
        from typing import cast

        cast(AsyncMock, mock_attack_strategy._perform_attack_async).return_value = mock_result

        with caplog.at_level(logging.INFO, logger=""):
            await mock_attack_strategy.execute_async(context=basic_context)
        assert any("did not achieve the objective" in record.message for record in caplog.records)
        assert any("Target refused" in record.message for record in caplog.records)

        # Test UNDETERMINED outcome
        caplog.clear()
        mock_result.outcome = AttackOutcome.UNDETERMINED
        mock_result.outcome_reason = None

        with caplog.at_level(logging.INFO, logger=""):
            await mock_attack_strategy.execute_async(context=basic_context)
        assert any("outcome is undetermined" in record.message for record in caplog.records)
        assert any("Not specified" in record.message for record in caplog.records)


@pytest.mark.usefixtures("patch_central_database")
class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling"""

    def test_cannot_instantiate_abstract_base_class(self):
        with pytest.raises(TypeError) as exc_info:
            AttackStrategy()  # type: ignore

        # Should mention the abstract methods
        error_msg = str(exc_info.value)
        assert "_validate_context" in error_msg or "abstract" in error_msg

    def test_memory_labels_invalid_format(self):
        with patch("pyrit.attacks.base.attack_strategy.default_values") as mock:
            mock.get_non_required_value.return_value = "invalid json"

            # ast.literal_eval will raise SyntaxError for invalid syntax
            with pytest.raises(SyntaxError):

                class ConcreteStrategy(AttackStrategy):
                    _validate_context = MagicMock()
                    _setup_async = AsyncMock()
                    _perform_attack_async = AsyncMock()
                    _teardown_async = AsyncMock()

                ConcreteStrategy()

    def test_multiple_instances_have_unique_ids(self, mock_default_values):
        class ConcreteStrategy(AttackStrategy):
            _validate_context = MagicMock()
            _setup_async = AsyncMock()
            _perform_attack_async = AsyncMock()
            _teardown_async = AsyncMock()

        strategy1 = ConcreteStrategy()
        strategy2 = ConcreteStrategy()

        assert strategy1._id != strategy2._id
        assert strategy1.get_identifier()["id"] != strategy2.get_identifier()["id"]
