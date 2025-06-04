# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.attacks.base.attack_strategy import AttackStrategy, AttackStrategyLogAdapter
from pyrit.exceptions.exception_classes import (
    AttackExecutionError,
    AttackValidationError,
)


@pytest.fixture
def basic_context():
    """Create a basic mock context for testing"""
    context = MagicMock()
    context.objective = "Test objective"
    context.memory_labels = {"test": "label"}
    return context


@pytest.fixture
def mock_memory():
    """Mock the CentralMemory instance"""
    with patch("pyrit.attacks.base.attack_strategy.CentralMemory") as mock:
        memory_instance = MagicMock()
        mock.get_memory_instance.return_value = memory_instance
        yield memory_instance


@pytest.fixture
def mock_default_values():
    """Mock default values for memory labels"""
    with patch("pyrit.attacks.base.attack_strategy.default_values") as mock:
        mock.get_non_required_value.return_value = '{"test_label": "test_value"}'
        yield mock


@pytest.fixture
def mock_attack_strategy(mock_memory, mock_default_values):
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

    return TestableAttackStrategy()


class TestAttackStrategyInitialization:
    """Tests for AttackStrategy initialization"""

    def test_init_with_default_parameters(self, mock_memory, mock_default_values):
        class ConcreteStrategy(AttackStrategy):
            _validate_context = MagicMock()
            _setup_async = AsyncMock()
            _perform_attack_async = AsyncMock()
            _teardown_async = AsyncMock()

        strategy = ConcreteStrategy()

        assert strategy._id is not None
        assert isinstance(strategy._id, uuid.UUID)
        assert strategy._memory == mock_memory
        assert strategy._memory_labels == {"test_label": "test_value"}
        assert isinstance(strategy._logger, AttackStrategyLogAdapter)

    def test_init_with_custom_logger(self, mock_memory, mock_default_values):
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
    def test_init_with_various_memory_labels(self, mock_memory, memory_labels_str, expected):
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


class TestAttackExecution:
    """Tests for the main attack execution logic (execute_async)"""

    @pytest.mark.asyncio
    async def test_execute_async_successful_flow(self, mock_attack_strategy, basic_context):
        mock_result = MagicMock()
        mock_attack_strategy._perform_attack_async.return_value = mock_result

        result = await mock_attack_strategy.execute_async(context=basic_context)

        # Verify all lifecycle methods were called
        mock_attack_strategy._validate_context.assert_called_once_with(context=basic_context)
        mock_attack_strategy._setup_async.assert_called_once_with(context=basic_context)
        mock_attack_strategy._perform_attack_async.assert_called_once_with(context=basic_context)
        mock_attack_strategy._teardown_async.assert_called_once_with(context=basic_context)

        # Verify result
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_execute_async_validation_failure(self, mock_attack_strategy, basic_context):
        mock_attack_strategy._validate_context.side_effect = ValueError("Validation failed")

        with pytest.raises(AttackValidationError) as exc_info:
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

        with pytest.raises(AttackExecutionError) as exc_info:
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

        with pytest.raises(AttackExecutionError):
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
        with pytest.raises(AttackExecutionError):
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
            AttackValidationError(message="Existing validation error"),
            AttackExecutionError(message="Existing execution error"),
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


class TestConcurrency:
    """Tests for concurrent execution scenarios"""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_executions(self, mock_memory, mock_default_values):
        # Create multiple independent strategy instances
        strategies = []
        for _ in range(5):

            class ConcreteStrategy(AttackStrategy):
                _validate_context = MagicMock()
                _setup_async = AsyncMock()
                _perform_attack_async = AsyncMock(return_value=MagicMock(success=True))
                _teardown_async = AsyncMock()

            strategies.append(ConcreteStrategy())

        # Create multiple mock contexts
        contexts = [MagicMock(objective=f"Objective {i}") for i in range(5)]

        # Execute concurrently
        tasks = [strategy.execute_async(context=ctx) for strategy, ctx in zip(strategies, contexts)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 5
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_execution_isolation(self, mock_attack_strategy):
        """Ensure concurrent executions don't interfere with each other"""
        # Create contexts with different objectives
        context1 = MagicMock(objective="Objective 1")
        context2 = MagicMock(objective="Objective 2")

        # Configure return values
        result1 = MagicMock(id="result1")
        result2 = MagicMock(id="result2")
        mock_attack_strategy._perform_attack_async.side_effect = [result1, result2]

        # Execute concurrently
        results = await asyncio.gather(
            mock_attack_strategy.execute_async(context=context1), mock_attack_strategy.execute_async(context=context2)
        )

        # Both should succeed with correct results
        assert len(results) == 2
        assert results[0].id == "result1"
        assert results[1].id == "result2"


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling"""

    def test_cannot_instantiate_abstract_base_class(self):
        with pytest.raises(TypeError) as exc_info:
            AttackStrategy()  # type: ignore

        # Should mention the abstract methods
        error_msg = str(exc_info.value)
        assert "_validate_context" in error_msg or "abstract" in error_msg

    def test_memory_labels_invalid_format(self, mock_memory):
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

    def test_multiple_instances_have_unique_ids(self, mock_memory, mock_default_values):
        class ConcreteStrategy(AttackStrategy):
            _validate_context = MagicMock()
            _setup_async = AsyncMock()
            _perform_attack_async = AsyncMock()
            _teardown_async = AsyncMock()

        strategy1 = ConcreteStrategy()
        strategy2 = ConcreteStrategy()

        assert strategy1._id != strategy2._id
        assert strategy1.get_identifier()["id"] != strategy2.get_identifier()["id"]
