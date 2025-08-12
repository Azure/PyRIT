# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.workflow.xpia import (
    XPIAContext,
    XPIAResult,
    XPIAStatus,
    XPIAWorkflow,
)
from pyrit.models import PromptRequestResponse, Score, SeedPrompt, SeedPromptGroup
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer


# Shared fixtures for all test classes
@pytest.fixture
def mock_attack_setup_target() -> MagicMock:
    """Create a mock attack setup target."""
    target = MagicMock(spec=PromptTarget)
    return target


@pytest.fixture
def mock_scorer() -> MagicMock:
    """Create a mock scorer."""
    scorer = MagicMock(spec=Scorer)
    scorer.score_text_async = AsyncMock()
    return scorer


@pytest.fixture
def mock_prompt_normalizer() -> MagicMock:
    """Create a mock prompt normalizer."""
    normalizer = MagicMock(spec=PromptNormalizer)
    normalizer.send_prompt_async = AsyncMock()
    return normalizer


@pytest.fixture
def valid_seed_prompt_group() -> SeedPromptGroup:
    """Create a valid seed prompt group for testing."""
    return SeedPromptGroup(prompts=[SeedPrompt(value="Test attack content", data_type="text", name="test_prompt")])


@pytest.fixture
def mock_processing_callback() -> AsyncMock:
    """Create a mock processing callback."""
    callback = AsyncMock()
    callback.return_value = "Processing response"
    return callback


@pytest.fixture
def valid_context(valid_seed_prompt_group: SeedPromptGroup, mock_processing_callback: AsyncMock) -> XPIAContext:
    """Create a valid XPIA context for testing."""
    return XPIAContext(
        attack_content=valid_seed_prompt_group,
        processing_callback=mock_processing_callback,
        memory_labels={"test": "label"},
    )


@pytest.fixture
def workflow(
    mock_attack_setup_target: MagicMock, mock_scorer: MagicMock, mock_prompt_normalizer: MagicMock
) -> XPIAWorkflow:
    """Create an XPIA workflow instance for testing."""
    return XPIAWorkflow(
        attack_setup_target=mock_attack_setup_target, scorer=mock_scorer, prompt_normalizer=mock_prompt_normalizer
    )


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowValidation:
    """Test class for XPIA workflow context validation."""

    def test_validate_context_with_valid_context(self, workflow: XPIAWorkflow, valid_context: XPIAContext) -> None:
        """Test that validation passes with a valid context."""
        # Should not raise any exception
        workflow._validate_context(context=valid_context)

    def test_validate_context_missing_attack_content_raises_error(
        self, workflow: XPIAWorkflow, mock_processing_callback: AsyncMock
    ) -> None:
        """Test that validation fails when attack_content is None."""
        context = XPIAContext(attack_content=None, processing_callback=mock_processing_callback)  # type: ignore

        with pytest.raises(ValueError, match="attack_content: SeedPromptGroup must be provided"):
            workflow._validate_context(context=context)

    def test_validate_context_empty_seed_prompt_group_raises_error(
        self, workflow: XPIAWorkflow, mock_processing_callback: AsyncMock
    ) -> None:
        """Test that validation fails when seed prompt group has invalid prompt."""
        context = XPIAContext(attack_content=None, processing_callback=mock_processing_callback)  # type: ignore

        with pytest.raises(ValueError, match="attack_content: SeedPromptGroup must be provided"):
            workflow._validate_context(context=context)

    def test_validate_context_multiple_prompts_raises_error(
        self, workflow: XPIAWorkflow, mock_processing_callback: AsyncMock
    ) -> None:
        """Test that validation fails when seed prompt group has multiple prompts."""
        multiple_prompts_group = SeedPromptGroup(
            prompts=[
                SeedPrompt(value="First prompt", data_type="text", name="prompt1"),
                SeedPrompt(value="Second prompt", data_type="text", name="prompt2"),
            ]
        )
        context = XPIAContext(attack_content=multiple_prompts_group, processing_callback=mock_processing_callback)

        with pytest.raises(ValueError, match="attack_content: Exactly one seed prompt must be provided"):
            workflow._validate_context(context=context)

    def test_validate_context_non_text_prompt_raises_error(
        self, workflow: XPIAWorkflow, mock_processing_callback: AsyncMock
    ) -> None:
        """Test that validation fails when prompt is not text type."""
        non_text_group = SeedPromptGroup(
            prompts=[SeedPrompt(value="image.jpg", data_type="image_path", name="image_prompt")]
        )
        context = XPIAContext(attack_content=non_text_group, processing_callback=mock_processing_callback)

        with pytest.raises(ValueError, match="attack_content: Prompt must be of type 'text'"):
            workflow._validate_context(context=context)

    def test_validate_context_missing_processing_callback_raises_error(
        self, workflow: XPIAWorkflow, valid_seed_prompt_group: SeedPromptGroup
    ) -> None:
        """Test that validation fails when processing_callback is None."""
        context = XPIAContext(attack_content=valid_seed_prompt_group, processing_callback=None)  # type: ignore

        with pytest.raises(ValueError, match="processing_callback is required"):
            workflow._validate_context(context=context)


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowPerform:
    """Test class for XPIA workflow perform method."""

    @pytest.mark.asyncio
    async def test_perform_async_complete_workflow_with_scorer(
        self,
        workflow: XPIAWorkflow,
        valid_seed_prompt_group: SeedPromptGroup,
        mock_prompt_normalizer: MagicMock,
        mock_scorer: MagicMock,
    ) -> None:
        """Test complete workflow execution with scorer."""
        # Create a specific mock processing callback for this test
        mock_processing_callback = AsyncMock()
        mock_processing_callback.return_value = "Processing response"

        # Create context with the mock callback
        context = XPIAContext(
            attack_content=valid_seed_prompt_group,
            processing_callback=mock_processing_callback,
            memory_labels={"test": "label"},
        )

        # Setup mock responses
        mock_response = MagicMock()
        mock_response.get_value.return_value = "Attack setup response"
        mock_prompt_normalizer.send_prompt_async.return_value = mock_response

        mock_score = MagicMock(spec=Score)
        mock_score.get_value.return_value = 0.8
        mock_scorer.score_text_async.return_value = [mock_score]

        # Execute workflow
        result = await workflow._perform_async(context=context)

        # Verify result
        assert isinstance(result, XPIAResult)
        assert result.processing_conversation_id == context.processing_conversation_id
        assert result.processing_response == "Processing response"
        assert result.score == mock_score
        assert result.attack_setup_response == "Attack setup response"

        # Verify method calls
        mock_prompt_normalizer.send_prompt_async.assert_called_once()
        mock_processing_callback.assert_called_once()
        mock_scorer.score_text_async.assert_called_once_with("Processing response")

    @pytest.mark.asyncio
    async def test_perform_async_workflow_without_scorer(
        self,
        mock_attack_setup_target: MagicMock,
        mock_prompt_normalizer: MagicMock,
        valid_seed_prompt_group: SeedPromptGroup,
    ) -> None:
        """Test workflow execution without scorer."""
        # Create a specific mock processing callback for this test
        mock_processing_callback = AsyncMock()
        mock_processing_callback.return_value = "Processing response"

        # Create context with the mock callback
        context = XPIAContext(
            attack_content=valid_seed_prompt_group,
            processing_callback=mock_processing_callback,
            memory_labels={"test": "label"},
        )

        # Create workflow without scorer
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target, scorer=None, prompt_normalizer=mock_prompt_normalizer
        )

        # Setup mock responses
        mock_response = MagicMock()
        mock_response.get_value.return_value = "Attack setup response"
        mock_prompt_normalizer.send_prompt_async.return_value = mock_response

        # Execute workflow
        result = await workflow._perform_async(context=context)

        # Verify result
        assert isinstance(result, XPIAResult)
        assert result.processing_conversation_id == context.processing_conversation_id
        assert result.processing_response == "Processing response"
        assert result.score is None
        assert result.attack_setup_response == "Attack setup response"

        # Verify method calls
        mock_prompt_normalizer.send_prompt_async.assert_called_once()
        mock_processing_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_async_scorer_returns_empty_list(
        self,
        workflow: XPIAWorkflow,
        valid_context: XPIAContext,
        mock_prompt_normalizer: MagicMock,
        mock_scorer: MagicMock,
    ) -> None:
        """Test workflow when scorer returns empty list."""
        # Setup mock responses
        mock_response = MagicMock()
        mock_response.get_value.return_value = "Attack setup response"
        mock_prompt_normalizer.send_prompt_async.return_value = mock_response

        mock_scorer.score_text_async.return_value = []

        # Execute workflow
        result = await workflow._perform_async(context=valid_context)

        # Verify result
        assert isinstance(result, XPIAResult)
        assert result.processing_conversation_id == valid_context.processing_conversation_id
        assert result.processing_response == "Processing response"
        assert result.score is None
        assert result.attack_setup_response == "Attack setup response"

    @pytest.mark.asyncio
    async def test_perform_async_scorer_raises_exception(
        self,
        workflow: XPIAWorkflow,
        valid_context: XPIAContext,
        mock_prompt_normalizer: MagicMock,
        mock_scorer: MagicMock,
    ) -> None:
        """Test workflow when scorer raises an exception."""
        # Setup mock responses
        mock_response = MagicMock()
        mock_response.get_value.return_value = "Attack setup response"
        mock_prompt_normalizer.send_prompt_async.return_value = mock_response

        mock_scorer.score_text_async.side_effect = Exception("Scoring error")

        # Execute workflow
        result = await workflow._perform_async(context=valid_context)

        # Verify result
        assert isinstance(result, XPIAResult)
        assert result.processing_conversation_id == valid_context.processing_conversation_id
        assert result.processing_response == "Processing response"
        assert result.score is None
        assert result.attack_setup_response == "Attack setup response"

    @pytest.mark.asyncio
    async def test_setup_attack_async_calls_prompt_normalizer_correctly(
        self, workflow: XPIAWorkflow, valid_context: XPIAContext, mock_prompt_normalizer: MagicMock
    ) -> None:
        """Test that setup attack calls prompt normalizer with correct parameters."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.get_value.return_value = "Attack setup response"
        mock_prompt_normalizer.send_prompt_async.return_value = mock_response

        # Execute setup attack
        response = await workflow._setup_attack_async(context=valid_context)

        # Verify response
        assert response == "Attack setup response"

        # Verify prompt normalizer was called with correct parameters
        call_args = mock_prompt_normalizer.send_prompt_async.call_args
        assert call_args.kwargs["seed_prompt_group"] == valid_context.attack_content
        assert call_args.kwargs["target"] == workflow._attack_setup_target
        assert call_args.kwargs["labels"] == valid_context.memory_labels
        assert call_args.kwargs["conversation_id"] == valid_context.attack_setup_target_conversation_id

    @pytest.mark.asyncio
    @patch("pyrit.executor.workflow.xpia.CentralMemory")
    async def test_execute_processing_async_adds_to_memory(
        self, mock_memory_class: MagicMock, workflow: XPIAWorkflow, valid_context: XPIAContext
    ) -> None:
        """Test that execute processing adds response to memory."""
        # Setup mock memory
        mock_memory_instance = MagicMock()
        mock_memory_class.get_memory_instance.return_value = mock_memory_instance

        # Patch the workflow's _memory attribute to use our mock
        workflow._memory = mock_memory_instance

        # Execute processing
        response = await workflow._execute_processing_async(context=valid_context)

        # Verify response
        assert response == "Processing response"

        # Verify memory addition
        mock_memory_instance.add_request_response_to_memory.assert_called_once()
        call_args = mock_memory_instance.add_request_response_to_memory.call_args
        assert call_args.kwargs["request"] is not None
        assert isinstance(call_args.kwargs["request"], PromptRequestResponse)

    @pytest.mark.asyncio
    async def test_score_response_async_with_no_scorer(
        self, mock_attack_setup_target: MagicMock, mock_prompt_normalizer: MagicMock
    ) -> None:
        """Test scoring when no scorer is provided."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target, scorer=None, prompt_normalizer=mock_prompt_normalizer
        )

        result = await workflow._score_response_async(processing_response="test response")

        assert result is None


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowExecution:
    """Test class for XPIA workflow execution methods."""

    @pytest.mark.asyncio
    async def test_execute_async_with_valid_parameters(
        self,
        mock_attack_setup_target: MagicMock,
        mock_scorer: MagicMock,
        valid_seed_prompt_group: SeedPromptGroup,
        mock_processing_callback: AsyncMock,
    ) -> None:
        """Test execute_async with valid parameters."""
        # Create workflow with mocked PromptNormalizer
        with patch("pyrit.executor.workflow.xpia.PromptNormalizer") as mock_normalizer_class:
            mock_normalizer = MagicMock()
            mock_normalizer.send_prompt_async = AsyncMock()
            mock_normalizer_class.return_value = mock_normalizer

            workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target, scorer=mock_scorer)

            with (
                patch.object(workflow, "_perform_async") as mock_perform,
                patch.object(workflow, "_validate_context") as mock_validate,
                patch.object(workflow, "_setup_async") as mock_setup,
                patch.object(workflow, "_teardown_async") as mock_teardown,
            ):

                # Setup mock return value
                expected_result = XPIAResult(
                    processing_conversation_id="test-conversation-id",
                    processing_response="test response",
                    score=None,
                    attack_setup_response="setup response",
                )
                mock_perform.return_value = expected_result

                # Execute workflow
                result = await workflow.execute_async(
                    attack_content=valid_seed_prompt_group,
                    processing_callback=mock_processing_callback,
                    memory_labels={"test": "label"},
                )

                # Verify result
                assert result == expected_result

                # Verify lifecycle methods were called
                mock_validate.assert_called_once()
                mock_setup.assert_called_once()
                mock_perform.assert_called_once()
                mock_teardown.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_async_invalid_attack_content_type_raises_error(
        self, mock_attack_setup_target: MagicMock, mock_scorer: MagicMock, mock_processing_callback: AsyncMock
    ) -> None:
        """Test that execute_async raises error with invalid attack_content type."""
        # Create workflow with mocked PromptNormalizer
        with patch("pyrit.executor.workflow.xpia.PromptNormalizer") as mock_normalizer_class:
            mock_normalizer = MagicMock()
            mock_normalizer.send_prompt_async = AsyncMock()
            mock_normalizer_class.return_value = mock_normalizer

            workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target, scorer=mock_scorer)

            with pytest.raises(TypeError):
                await workflow.execute_async(
                    attack_content="invalid_type",  # Should be SeedPromptGroup
                    processing_callback=mock_processing_callback,
                )

    @pytest.mark.asyncio
    async def test_execute_async_invalid_processing_callback_type_raises_error(
        self, mock_attack_setup_target: MagicMock, mock_scorer: MagicMock, valid_seed_prompt_group: SeedPromptGroup
    ) -> None:
        """Test that execute_async raises error with invalid processing_callback type."""
        # Create workflow with mocked PromptNormalizer
        with patch("pyrit.executor.workflow.xpia.PromptNormalizer") as mock_normalizer_class:
            mock_normalizer = MagicMock()
            mock_normalizer.send_prompt_async = AsyncMock()
            mock_normalizer_class.return_value = mock_normalizer

            workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target, scorer=mock_scorer)

            with pytest.raises(TypeError, match="processing_callback must be callable"):
                await workflow.execute_async(
                    attack_content=valid_seed_prompt_group, processing_callback="not_callable"  # Should be callable
                )

    @pytest.mark.asyncio
    async def test_execute_async_invalid_memory_labels_type_raises_error(
        self,
        mock_attack_setup_target: MagicMock,
        mock_scorer: MagicMock,
        valid_seed_prompt_group: SeedPromptGroup,
        mock_processing_callback: AsyncMock,
    ) -> None:
        """Test that execute_async raises error with invalid memory_labels type."""
        # Create workflow with mocked PromptNormalizer
        with patch("pyrit.executor.workflow.xpia.PromptNormalizer") as mock_normalizer_class:
            mock_normalizer = MagicMock()
            mock_normalizer.send_prompt_async = AsyncMock()
            mock_normalizer_class.return_value = mock_normalizer

            workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target, scorer=mock_scorer)

            with pytest.raises(TypeError):
                await workflow.execute_async(
                    attack_content=valid_seed_prompt_group,
                    processing_callback=mock_processing_callback,
                    memory_labels="invalid_type",  # Should be dict
                )

    @pytest.mark.asyncio
    async def test_execute_async_missing_required_attack_content_raises_error(
        self, mock_attack_setup_target: MagicMock, mock_scorer: MagicMock, mock_processing_callback: AsyncMock
    ) -> None:
        """Test that execute_async raises error when attack_content is missing."""
        # Create workflow with mocked PromptNormalizer
        with patch("pyrit.executor.workflow.xpia.PromptNormalizer") as mock_normalizer_class:
            mock_normalizer = MagicMock()
            mock_normalizer.send_prompt_async = AsyncMock()
            mock_normalizer_class.return_value = mock_normalizer

            workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target, scorer=mock_scorer)

            with pytest.raises(ValueError):
                await workflow.execute_async(
                    processing_callback=mock_processing_callback
                    # attack_content is required but missing
                )

    @pytest.mark.asyncio
    async def test_setup_async_generates_conversation_ids(
        self, workflow: XPIAWorkflow, valid_seed_prompt_group: SeedPromptGroup, mock_processing_callback: AsyncMock
    ) -> None:
        """Test that setup_async generates conversation IDs and combines memory labels."""
        context = XPIAContext(
            attack_content=valid_seed_prompt_group,
            processing_callback=mock_processing_callback,
            memory_labels={"test": "label"},
        )

        # Store original IDs to verify they change
        original_attack_id = context.attack_setup_target_conversation_id
        original_processing_id = context.processing_conversation_id

        await workflow._setup_async(context=context)

        # Verify IDs were regenerated
        assert context.attack_setup_target_conversation_id != original_attack_id
        assert context.processing_conversation_id != original_processing_id

        # Verify UUIDs are valid
        uuid.UUID(context.attack_setup_target_conversation_id)
        uuid.UUID(context.processing_conversation_id)

        # Verify memory labels were combined
        assert "test" in context.memory_labels

    @pytest.mark.asyncio
    async def test_teardown_async_completes_successfully(
        self, workflow: XPIAWorkflow, valid_seed_prompt_group: SeedPromptGroup, mock_processing_callback: AsyncMock
    ) -> None:
        """Test that teardown_async completes without errors."""
        context = XPIAContext(attack_content=valid_seed_prompt_group, processing_callback=mock_processing_callback)

        # Should not raise any exception
        await workflow._teardown_async(context=context)


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAResult:
    """Test class for XPIAResult properties."""

    def test_success_property_with_positive_score(self) -> None:
        """Test success property returns True for positive score."""
        mock_score = MagicMock(spec=Score)
        mock_score.get_value.return_value = 0.8

        result = XPIAResult(processing_conversation_id="test-id", processing_response="test response", score=mock_score)

        assert result.success is True

    def test_success_property_with_zero_score(self) -> None:
        """Test success property returns False for zero score."""
        mock_score = MagicMock(spec=Score)
        mock_score.get_value.return_value = 0.0

        result = XPIAResult(processing_conversation_id="test-id", processing_response="test response", score=mock_score)

        assert result.success is False

    def test_success_property_with_negative_score(self) -> None:
        """Test success property returns False for negative score."""
        mock_score = MagicMock(spec=Score)
        mock_score.get_value.return_value = -0.5

        result = XPIAResult(processing_conversation_id="test-id", processing_response="test response", score=mock_score)

        assert result.success is False

    def test_success_property_with_no_score(self) -> None:
        """Test success property returns False when no score is provided."""
        result = XPIAResult(processing_conversation_id="test-id", processing_response="test response", score=None)

        assert result.success is False

    def test_success_property_with_non_numeric_score(self) -> None:
        """Test success property returns False for non-numeric score."""
        mock_score = MagicMock(spec=Score)
        mock_score.get_value.return_value = "invalid"

        result = XPIAResult(processing_conversation_id="test-id", processing_response="test response", score=mock_score)

        assert result.success is False

    def test_status_property_success(self) -> None:
        """Test status property returns SUCCESS for successful attack."""
        mock_score = MagicMock(spec=Score)
        mock_score.get_value.return_value = 0.8

        result = XPIAResult(processing_conversation_id="test-id", processing_response="test response", score=mock_score)

        assert result.status == XPIAStatus.SUCCESS

    def test_status_property_failure(self) -> None:
        """Test status property returns FAILURE for failed attack."""
        mock_score = MagicMock(spec=Score)
        mock_score.get_value.return_value = 0.0

        result = XPIAResult(processing_conversation_id="test-id", processing_response="test response", score=mock_score)

        assert result.status == XPIAStatus.FAILURE

    def test_status_property_unknown(self) -> None:
        """Test status property returns UNKNOWN when no score is provided."""
        result = XPIAResult(processing_conversation_id="test-id", processing_response="test response", score=None)

        assert result.status == XPIAStatus.UNKNOWN
