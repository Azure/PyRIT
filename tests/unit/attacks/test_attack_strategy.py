# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
import uuid
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.attacks.base.attack_strategy import AttackStrategy, AttackStrategyLogAdapter
from pyrit.exceptions.exception_classes import (
    AttackExecutionException,
    AttackValidationException,
)
from pyrit.models import AttackOutcome


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

    # Create a mock context type
    mock_context_type = MagicMock()
    mock_context_type.create_from_params = MagicMock()

    # Create a concrete subclass with mocked abstract methods
    class TestableAttackStrategy(AttackStrategy):
        def __init__(self, **kwargs):
            # Use the root logger to ensure caplog can capture the logs
            super().__init__(context_type=mock_context_type, logger=logging.getLogger(), **kwargs)

        # Mock all abstract methods
        _validate_context = MagicMock()
        _setup_async = AsyncMock()
        _perform_attack_async = AsyncMock()
        _teardown_async = AsyncMock()

    strategy = TestableAttackStrategy()

    # Mock the memory to avoid any issues with the real memory system
    mock_memory = MagicMock()
    strategy._memory = mock_memory

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
        mock_context_type = MagicMock()

        class ConcreteStrategy(AttackStrategy):
            _validate_context = MagicMock()
            _setup_async = AsyncMock()
            _perform_attack_async = AsyncMock()
            _teardown_async = AsyncMock()

        strategy = ConcreteStrategy(context_type=mock_context_type)

        assert strategy._id is not None
        assert isinstance(strategy._id, uuid.UUID)
        assert strategy._memory is not None
        assert strategy._memory_labels == {"test_label": "test_value"}
        assert isinstance(strategy._logger, AttackStrategyLogAdapter)
        assert strategy._context_type == mock_context_type

    def test_init_with_custom_logger(self, mock_default_values):
        mock_context_type = MagicMock()

        class ConcreteStrategy(AttackStrategy):
            _validate_context = MagicMock()
            _setup_async = AsyncMock()
            _perform_attack_async = AsyncMock()
            _teardown_async = AsyncMock()

        custom_logger = logging.getLogger("custom_test_logger")
        strategy = ConcreteStrategy(context_type=mock_context_type, logger=custom_logger)

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
            mock_context_type = MagicMock()

            class ConcreteStrategy(AttackStrategy):
                _validate_context = MagicMock()
                _setup_async = AsyncMock()
                _perform_attack_async = AsyncMock()
                _teardown_async = AsyncMock()

            strategy = ConcreteStrategy(context_type=mock_context_type)
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
        # Configure context type's create_from_params to return the basic_context
        mock_attack_strategy._context_type.create_from_params.return_value = basic_context

        result = await mock_attack_strategy.execute_async(objective="Test objective")

        # Verify context creation was called
        mock_attack_strategy._context_type.create_from_params.assert_called_once()

        # Verify all lifecycle methods were called
        mock_attack_strategy._validate_context.assert_called_once_with(context=basic_context)
        mock_attack_strategy._setup_async.assert_called_once_with(context=basic_context)
        mock_attack_strategy._perform_attack_async.assert_called_once_with(context=basic_context)
        mock_attack_strategy._memory.add_attack_results_to_memory.assert_called_once()
        mock_attack_strategy._teardown_async.assert_called_once_with(context=basic_context)

        # Verify result
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.outcome_reason == "Test successful"
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_async_validation_failure(self, mock_attack_strategy, basic_context):
        mock_attack_strategy._context_type.create_from_params.return_value = basic_context
        mock_attack_strategy._validate_context.side_effect = ValueError("Validation failed")

        with pytest.raises(AttackValidationException) as exc_info:
            await mock_attack_strategy.execute_async(objective="Test objective")

        # Verify error details
        assert "Context validation failed" in str(exc_info.value)
        assert exc_info.value.context_info["attack_type"] == "TestableAttackStrategy"
        assert exc_info.value.context_info["original_error"] == "Validation failed"

        # Verify lifecycle - only validation should be called
        mock_attack_strategy._validate_context.assert_called_once()
        mock_attack_strategy._setup_async.assert_not_called()
        mock_attack_strategy._perform_attack_async.assert_not_called()
        mock_attack_strategy._teardown_async.assert_not_called()
        mock_attack_strategy._memory.add_attack_results_to_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_async_setup_failure_calls_teardown(self, mock_attack_strategy, basic_context):
        mock_attack_strategy._context_type.create_from_params.return_value = basic_context
        mock_attack_strategy._setup_async.side_effect = RuntimeError("Setup failed")

        with pytest.raises(AttackExecutionException) as exc_info:
            await mock_attack_strategy.execute_async(objective="Test objective")

        # Verify error details
        assert "Unexpected error during attack execution" in str(exc_info.value)
        assert exc_info.value.attack_name == "TestableAttackStrategy"
        assert exc_info.value.objective == "Test objective"

        # Verify lifecycle - teardown should still be called, but memory should not be called
        mock_attack_strategy._validate_context.assert_called_once()
        mock_attack_strategy._setup_async.assert_called_once()
        mock_attack_strategy._perform_attack_async.assert_not_called()
        mock_attack_strategy._teardown_async.assert_called_once()
        mock_attack_strategy._memory.add_attack_results_to_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_async_perform_failure_calls_teardown(self, mock_attack_strategy, basic_context):
        mock_attack_strategy._context_type.create_from_params.return_value = basic_context
        mock_attack_strategy._perform_attack_async.side_effect = RuntimeError("Attack failed")

        with pytest.raises(AttackExecutionException):
            await mock_attack_strategy.execute_async(objective="Test objective")

        # Verify lifecycle - teardown should still be called, but memory should not be called since attack failed
        mock_attack_strategy._validate_context.assert_called_once()
        mock_attack_strategy._setup_async.assert_called_once()
        mock_attack_strategy._perform_attack_async.assert_called_once()
        mock_attack_strategy._teardown_async.assert_called_once()
        mock_attack_strategy._memory.add_attack_results_to_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_async_teardown_failure_propagates(self, mock_attack_strategy, basic_context):
        mock_attack_strategy._context_type.create_from_params.return_value = basic_context
        mock_attack_strategy._teardown_async.side_effect = RuntimeError("Teardown failed")

        # Teardown failures should still propagate but after being called
        with pytest.raises(AttackExecutionException):
            await mock_attack_strategy.execute_async(objective="Test objective")

        # All methods should have been called, including memory since attack completed
        mock_attack_strategy._validate_context.assert_called_once()
        mock_attack_strategy._setup_async.assert_called_once()
        mock_attack_strategy._perform_attack_async.assert_called_once()
        mock_attack_strategy._teardown_async.assert_called_once()
        mock_attack_strategy._memory.add_attack_results_to_memory.assert_called_once()

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
        mock_attack_strategy._context_type.create_from_params.return_value = basic_context
        mock_attack_strategy._perform_attack_async.side_effect = existing_exception

        with pytest.raises(type(existing_exception)) as exc_info:
            await mock_attack_strategy.execute_async(objective="Test objective")

        # Should preserve the original exception type
        assert exc_info.value is existing_exception

    @pytest.mark.asyncio
    async def test_execute_async_sets_execution_time(self, mock_attack_strategy, basic_context):
        """Test that execute_async properly sets execution_time_ms on the result"""
        mock_attack_strategy._context_type.create_from_params.return_value = basic_context

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
        result = await mock_attack_strategy.execute_async(objective="Test objective")

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
        mock_attack_strategy._context_type.create_from_params.return_value = basic_context

        # Configure mock result with a pre-existing execution time
        mock_result = MagicMock()
        mock_result.outcome = AttackOutcome.SUCCESS
        mock_result.outcome_reason = "Test successful"
        mock_result.execution_time_ms = 999999  # Pre-existing value that should be overwritten

        from typing import cast

        cast(AsyncMock, mock_attack_strategy._perform_attack_async).return_value = mock_result

        # Execute the attack
        result = await mock_attack_strategy.execute_async(objective="Test objective")

        # Verify execution time was overwritten
        assert result.execution_time_ms is not None
        assert result.execution_time_ms != 999999  # Should have been overwritten
        assert result.execution_time_ms >= 0  # Should be a valid time measurement

    @pytest.mark.asyncio
    async def test_execute_with_context_async_works(self, mock_attack_strategy, basic_context):
        """Test that execute_with_context_async works for backward compatibility"""
        result = await mock_attack_strategy.execute_with_context_async(context=basic_context)

        # Verify all lifecycle methods were called
        mock_attack_strategy._validate_context.assert_called_once_with(context=basic_context)
        mock_attack_strategy._memory.add_attack_results_to_memory.assert_called_once()
        mock_attack_strategy._setup_async.assert_called_once_with(context=basic_context)
        mock_attack_strategy._perform_attack_async.assert_called_once_with(context=basic_context)
        mock_attack_strategy._teardown_async.assert_called_once_with(context=basic_context)

        # Verify result
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.outcome_reason == "Test successful"
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_async_with_all_parameters(self, mock_attack_strategy, basic_context):
        """Test execute_async with all optional parameters"""
        prepended_conv = [MagicMock()]
        memory_labels = {"custom": "label"}

        mock_attack_strategy._context_type.create_from_params.return_value = basic_context

        result = await mock_attack_strategy.execute_async(
            objective="Test objective",
            prepended_conversation=prepended_conv,
            memory_labels=memory_labels,
            custom_param="custom_value",
        )

        # Verify context type's create_from_params was called with all parameters
        mock_attack_strategy._context_type.create_from_params.assert_called_once_with(
            objective="Test objective",
            prepended_conversation=prepended_conv,
            memory_labels=memory_labels,
            custom_param="custom_value",
        )

        assert result.outcome == AttackOutcome.SUCCESS


@pytest.mark.usefixtures("patch_central_database")
class TestLogging:
    """Tests for logging functionality"""

    @pytest.mark.asyncio
    async def test_logging_during_lifecycle(self, mock_attack_strategy, basic_context, caplog):
        mock_attack_strategy._context_type.create_from_params.return_value = basic_context

        # Ensure we're capturing at the root logger level
        with caplog.at_level(logging.DEBUG, logger=""):
            await mock_attack_strategy.execute_async(objective="Test objective")

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
        for i in range(5):
            mock_context_type = MagicMock()

            class ConcreteStrategy(AttackStrategy):
                _validate_context = MagicMock()
                _setup_async = AsyncMock()
                _perform_attack_async = AsyncMock()
                _teardown_async = AsyncMock()

            strategy = ConcreteStrategy(context_type=mock_context_type)

            # Mock the memory to avoid any issues with the real memory system
            mock_memory = MagicMock()
            strategy._memory = mock_memory

            # Configure mock result with required attributes
            mock_result = MagicMock()
            mock_result.outcome = AttackOutcome.SUCCESS
            mock_result.outcome_reason = "Test successful"
            mock_result.execution_time_ms = None

            cast(AsyncMock, strategy._perform_attack_async).return_value = mock_result

            # Configure context creation
            mock_context = MagicMock(objective=f"Objective {i}")

            strategy._context_type.create_from_params.return_value = mock_context

            strategies.append(strategy)

        # Execute concurrently
        tasks = [strategy.execute_async(objective=f"Objective {i}") for i, strategy in enumerate(strategies)]
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

        # Configure context creation to return different contexts
        mock_attack_strategy._context_type.create_from_params.side_effect = [context1, context2]

        # Cast to AsyncMock to satisfy type checker
        from typing import cast

        cast(AsyncMock, mock_attack_strategy._perform_attack_async).side_effect = [result1, result2]

        # Execute concurrently
        results = await asyncio.gather(
            mock_attack_strategy.execute_async(objective="Objective 1"),
            mock_attack_strategy.execute_async(objective="Objective 2"),
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
        mock_attack_strategy._context_type.create_from_params.return_value = basic_context

        # Test SUCCESS outcome
        with caplog.at_level(logging.INFO, logger=""):
            await mock_attack_strategy.execute_async(objective="Test objective")
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
            await mock_attack_strategy.execute_async(objective="Test objective")
        assert any("did not achieve the objective" in record.message for record in caplog.records)
        assert any("Target refused" in record.message for record in caplog.records)

        # Test UNDETERMINED outcome
        caplog.clear()
        mock_result.outcome = AttackOutcome.UNDETERMINED
        mock_result.outcome_reason = None

        with caplog.at_level(logging.INFO, logger=""):
            await mock_attack_strategy.execute_async(objective="Test objective")
        assert any("outcome is undetermined" in record.message for record in caplog.records)
        assert any("Not specified" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_execute_async_deep_copies_prepended_conversation(self, mock_attack_strategy, basic_context):
        """Test that execute_async deep copies prepended_conversation to avoid modifying the original"""
        # Create original prepended conversation with mutable objects
        original_piece = MagicMock()
        original_piece.original_value = "Original message"
        original_piece.labels = {"original": "label"}
        original_piece.prompt_metadata = {"key": "value"}

        original_response = MagicMock()
        original_response.request_pieces = [original_piece]

        prepended_conversation = [original_response]

        # Store references to verify they're not modified
        original_piece_id = id(original_piece)
        original_response_id = id(original_response)
        original_labels_id = id(original_piece.labels)
        original_metadata_id = id(original_piece.prompt_metadata)

        # Configure context type to capture the passed conversation
        captured_conversation = None

        def capture_conversation(**kwargs):
            nonlocal captured_conversation
            captured_conversation = kwargs.get("prepended_conversation")
            return basic_context

        mock_attack_strategy._context_type.create_from_params.side_effect = capture_conversation

        # Execute the attack
        await mock_attack_strategy.execute_async(
            objective="Test objective", prepended_conversation=prepended_conversation
        )

        # Verify deep copy was made
        assert captured_conversation is not None
        assert captured_conversation is not prepended_conversation  # Different list object
        assert len(captured_conversation) == 1

        # Verify the conversation was deep copied (different object references)
        copied_response = captured_conversation[0]
        assert id(copied_response) != original_response_id

        # Check that it's a different object instance
        assert copied_response is not original_response

        # Verify original objects weren't modified
        assert original_piece.original_value == "Original message"
        assert original_piece.labels == {"original": "label"}
        assert original_piece.prompt_metadata == {"key": "value"}
        assert id(original_piece) == original_piece_id
        assert id(original_piece.labels) == original_labels_id
        assert id(original_piece.prompt_metadata) == original_metadata_id

    @pytest.mark.asyncio
    async def test_execute_async_handles_none_prepended_conversation(self, mock_attack_strategy, basic_context):
        """Test that execute_async handles None prepended_conversation correctly"""
        # Configure context type to capture the passed conversation
        captured_conversation = None

        def capture_conversation(**kwargs):
            nonlocal captured_conversation
            captured_conversation = kwargs.get("prepended_conversation")
            return basic_context

        mock_attack_strategy._context_type.create_from_params.side_effect = capture_conversation

        # Execute with None prepended_conversation
        await mock_attack_strategy.execute_async(objective="Test objective", prepended_conversation=None)

        # Verify empty list was passed
        assert captured_conversation == []

    @pytest.mark.asyncio
    async def test_execute_async_deep_copy_with_complex_structure(self, mock_attack_strategy, basic_context):
        """Test deep copy with more complex prepended conversation structure"""
        # Create a more complex conversation structure
        piece1 = MagicMock()
        piece1.labels = {"label1": "value1", "nested": {"key": "value"}}
        piece1.prompt_metadata = {"meta1": ["item1", "item2"]}

        piece2 = MagicMock()
        piece2.labels = {"label2": "value2"}
        piece2.prompt_metadata = {"meta2": {"nested": "data"}}

        response1 = MagicMock()
        response1.request_pieces = [piece1, piece2]

        response2 = MagicMock()
        response2.request_pieces = [MagicMock()]

        prepended_conversation = [response1, response2]

        # Capture the conversation passed to create_from_params
        captured_args = None

        def capture_args(**kwargs):
            nonlocal captured_args
            captured_args = kwargs
            return basic_context

        mock_attack_strategy._context_type.create_from_params.side_effect = capture_args

        # Execute the attack
        await mock_attack_strategy.execute_async(
            objective="Test objective", prepended_conversation=prepended_conversation
        )

        # Verify the conversation was passed but as a different object
        assert captured_args is not None
        assert "prepended_conversation" in captured_args
        passed_conversation = captured_args["prepended_conversation"]

        # Verify it's a different object (deep copy)
        assert passed_conversation is not prepended_conversation
        assert len(passed_conversation) == 2

        # The responses should be different objects
        assert passed_conversation[0] is not response1
        assert passed_conversation[1] is not response2


@pytest.mark.usefixtures("patch_central_database")
class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling"""

    def test_cannot_instantiate_abstract_base_class(self):
        with pytest.raises(TypeError) as exc_info:
            AttackStrategy(context_type=MagicMock())  # type: ignore

        # Should mention the abstract methods
        error_msg = str(exc_info.value)
        assert "_validate_context" in error_msg or "abstract" in error_msg

    def test_memory_labels_invalid_format(self):
        with patch("pyrit.attacks.base.attack_strategy.default_values") as mock:
            mock.get_non_required_value.return_value = "invalid json"

            # ast.literal_eval will raise SyntaxError for invalid syntax
            with pytest.raises(SyntaxError):
                mock_context_type = MagicMock()

                class ConcreteStrategy(AttackStrategy):
                    _validate_context = MagicMock()
                    _setup_async = AsyncMock()
                    _perform_attack_async = AsyncMock()
                    _teardown_async = AsyncMock()

                ConcreteStrategy(context_type=mock_context_type)

    def test_multiple_instances_have_unique_ids(self, mock_default_values):
        mock_context_type = MagicMock()

        class ConcreteStrategy(AttackStrategy):
            _validate_context = MagicMock()
            _setup_async = AsyncMock()
            _perform_attack_async = AsyncMock()
            _teardown_async = AsyncMock()

        strategy1 = ConcreteStrategy(context_type=mock_context_type)
        strategy2 = ConcreteStrategy(context_type=mock_context_type)

        assert strategy1._id != strategy2._id
        assert strategy1.get_identifier()["id"] != strategy2.get_identifier()["id"]


@pytest.mark.usefixtures("patch_central_database")
class TestWarnIfSet:
    """Tests for the _warn_if_set utility method"""

    def test_warn_if_set_with_none_values(self, mock_attack_strategy, caplog):
        """Test that None values don't trigger warnings"""
        # Create a mock config with None values
        config = MagicMock()
        config.__class__.__name__ = "TestConfig"
        config.field1 = None
        config.field2 = None

        with caplog.at_level(logging.WARNING):
            mock_attack_strategy._warn_if_set(config=config, unused_fields=["field1", "field2"])

        # Should not have any warnings
        assert len(caplog.records) == 0

    def test_warn_if_set_with_set_values(self, mock_attack_strategy, caplog):
        """Test that set values trigger warnings"""
        # Create a mock config with set values
        config = MagicMock()
        config.__class__.__name__ = "TestConfig"
        config.field1 = "some_value"
        config.field2 = 42

        with caplog.at_level(logging.WARNING):
            mock_attack_strategy._warn_if_set(config=config, unused_fields=["field1", "field2"])

        # Should have warnings for both fields
        assert len(caplog.records) == 2
        assert (
            "field1 was provided in TestConfig but is not used by TestableAttackStrategy" in caplog.records[0].message
        )
        assert (
            "field2 was provided in TestConfig but is not used by TestableAttackStrategy" in caplog.records[1].message
        )

    def test_warn_if_set_with_empty_collections(self, mock_attack_strategy, caplog):
        """Test that empty collections don't trigger warnings"""
        # Create a mock config with empty collections
        config = MagicMock()
        config.__class__.__name__ = "TestConfig"
        config.empty_list = []
        config.empty_dict = {}
        config.empty_tuple = ()
        config.empty_string = ""

        with caplog.at_level(logging.WARNING):
            mock_attack_strategy._warn_if_set(
                config=config, unused_fields=["empty_list", "empty_dict", "empty_tuple", "empty_string"]
            )

        # Should not have any warnings
        assert len(caplog.records) == 0

    def test_warn_if_set_with_non_empty_collections(self, mock_attack_strategy, caplog):
        """Test that non-empty collections trigger warnings"""
        # Create a mock config with non-empty collections
        config = MagicMock()
        config.__class__.__name__ = "TestConfig"
        config.some_list = [1, 2, 3]
        config.some_dict = {"key": "value"}
        config.some_tuple = (1, 2)
        config.some_string = "hello"

        with caplog.at_level(logging.WARNING):
            mock_attack_strategy._warn_if_set(
                config=config, unused_fields=["some_list", "some_dict", "some_tuple", "some_string"]
            )

        # Should have warnings for all fields
        assert len(caplog.records) == 4
        for record in caplog.records:
            assert "was provided in TestConfig but is not used by TestableAttackStrategy" in record.message

    def test_warn_if_set_with_missing_fields(self, mock_attack_strategy, caplog):
        """Test handling of fields that don't exist in the config"""
        # Create a mock config without certain fields
        config = MagicMock()
        config.__class__.__name__ = "TestConfig"
        config.existing_field = "value"
        # Note: missing_field is not set on config

        # Remove the missing_field attribute to ensure it doesn't exist
        if hasattr(config, "missing_field"):
            delattr(config, "missing_field")

        with caplog.at_level(logging.WARNING):
            mock_attack_strategy._warn_if_set(config=config, unused_fields=["existing_field", "missing_field"])

        # Should have one warning for existing_field and one for missing_field not existing
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) == 2

        # Check for the specific warning messages
        messages = [r.message for r in warning_records]
        assert any("existing_field was provided in TestConfig" in msg for msg in messages)
        assert any("Field 'missing_field' does not exist in TestConfig" in msg for msg in messages)

    def test_warn_if_set_mixed_values(self, mock_attack_strategy, caplog):
        """Test with a mix of set and unset values"""
        # Create a mock config with mixed values
        config = MagicMock()
        config.__class__.__name__ = "MixedConfig"
        config.set_string = "value"
        config.none_value = None
        config.empty_list = []
        config.full_list = [1, 2, 3]
        config.zero_value = 0  # Should still trigger warning (0 is not None)
        config.false_value = False  # Should still trigger warning (False is not None)

        with caplog.at_level(logging.WARNING):
            mock_attack_strategy._warn_if_set(
                config=config,
                unused_fields=["set_string", "none_value", "empty_list", "full_list", "zero_value", "false_value"],
            )

        # Should have warnings for: set_string, full_list, zero_value, false_value
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) == 4

        # Check that the right fields triggered warnings
        warning_messages = " ".join(r.message for r in warning_records)
        assert "set_string" in warning_messages
        assert "full_list" in warning_messages
        assert "zero_value" in warning_messages
        assert "false_value" in warning_messages

        # These should not be in warnings
        assert "none_value" not in warning_messages
        assert "empty_list" not in warning_messages

    def test_warn_if_set_with_custom_objects(self, mock_attack_strategy, caplog):
        """Test with custom objects that have __len__ method"""

        # Create a custom object with __len__
        class CustomCollection:
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

        config = MagicMock()
        config.__class__.__name__ = "CustomConfig"
        config.empty_custom = CustomCollection([])
        config.full_custom = CustomCollection([1, 2, 3])

        with caplog.at_level(logging.WARNING):
            mock_attack_strategy._warn_if_set(config=config, unused_fields=["empty_custom", "full_custom"])

        # Should only warn about full_custom
        assert len(caplog.records) == 1
        assert "full_custom was provided in CustomConfig" in caplog.records[0].message

    def test_warn_if_set_empty_unused_fields_list(self, mock_attack_strategy, caplog):
        """Test that empty unused_fields list produces no warnings"""
        config = MagicMock()
        config.__class__.__name__ = "TestConfig"
        config.field1 = "value"

        with caplog.at_level(logging.WARNING):
            mock_attack_strategy._warn_if_set(config=config, unused_fields=[])

        # Should not have any warnings
        assert len(caplog.records) == 0

    @pytest.mark.parametrize(
        "value,should_warn",
        [
            (None, False),
            ("", False),
            ([], False),
            ({}, False),
            ((), False),
            (0, True),
            (False, True),
            (True, True),
            ("value", True),
            ([1], True),
            ({"a": 1}, True),
            ((1,), True),
        ],
    )
    def test_warn_if_set_various_value_types(self, mock_attack_strategy, caplog, value, should_warn):
        """Test warning behavior with various value types"""
        config = MagicMock()
        config.__class__.__name__ = "TestConfig"
        config.test_field = value

        with caplog.at_level(logging.WARNING):
            mock_attack_strategy._warn_if_set(config=config, unused_fields=["test_field"])

        if should_warn:
            assert len(caplog.records) == 1
            assert "test_field was provided in TestConfig" in caplog.records[0].message
        else:
            assert len(caplog.records) == 0
