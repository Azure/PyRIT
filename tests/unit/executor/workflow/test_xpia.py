# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.core import StrategyConverterConfig
from pyrit.executor.workflow.xpia import (
    XPIAContext,
    XPIAManualProcessingWorkflow,
    XPIAResult,
    XPIATestWorkflow,
    XPIAWorkflow,
)
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer


@pytest.fixture
def mock_attack_setup_target():
    """Create a mock attack setup target for testing"""
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = {"id": "mock_attack_setup_target_id"}
    return target


@pytest.fixture
def mock_processing_target():
    """Create a mock processing target for testing"""
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = {"id": "mock_processing_target_id"}
    return target


@pytest.fixture
def mock_scorer():
    """Create a mock scorer for testing"""
    scorer = MagicMock(spec=Scorer)
    scorer.score_text_async = AsyncMock()
    return scorer


@pytest.fixture
def mock_prompt_normalizer():
    """Create a mock prompt normalizer for testing"""
    normalizer = MagicMock(spec=PromptNormalizer)
    normalizer.send_prompt_async = AsyncMock()
    return normalizer


@pytest.fixture
def mock_converter_config():
    """Create a mock converter configuration for testing"""
    config = MagicMock(spec=StrategyConverterConfig)
    config.request_converters = []
    config.response_converters = []
    return config


@pytest.fixture
def basic_context():
    """Create a basic XPIA context for testing"""
    return XPIAContext(
        attack_content="Test jailbreak content",
        processing_callback=AsyncMock(return_value="Processing response"),
    )


@pytest.fixture
def sample_attack_response():
    """Create a sample attack setup response for testing"""
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value="Attack setup response",
                original_value_data_type="text",
            )
        ]
    )


@pytest.fixture
def sample_processing_response():
    """Create a sample processing response for testing"""
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value="Processing target response",
                original_value_data_type="text",
            )
        ]
    )


@pytest.fixture
def success_score():
    """Create a success score for testing"""
    return Score(
        score_type="true_false",
        score_value="true",
        score_category="xpia_success",
        score_value_description="XPIA attack was successful",
        score_rationale="Attack successfully bypassed defenses",
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
    )


@pytest.fixture
def failure_score():
    """Create a failure score for testing"""
    return Score(
        score_type="true_false",
        score_value="false",
        score_category="xpia_success",
        score_value_description="XPIA attack failed",
        score_rationale="Attack was blocked by defenses",
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
    )


@pytest.fixture
def float_score():
    """Create a float score for testing"""
    return Score(
        score_type="float_scale",
        score_value="0.85",
        score_category="xpia_effectiveness",
        score_value_description="Attack effectiveness rating",
        score_rationale="High effectiveness rating",
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
    )


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowContextValidation:
    """Tests for context validation logic."""

    def test_validate_context_with_valid_context(self, mock_attack_setup_target, basic_context):
        """Test that valid context passes validation without errors."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        # Should not raise any exception
        workflow._validate_context(context=basic_context)

    def test_validate_context_raises_error_for_empty_attack_content(self, mock_attack_setup_target):
        """Test validation fails with empty attack content."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        context = XPIAContext(
            attack_content="",  # Empty attack content
            processing_callback=AsyncMock(return_value="test response"),
        )

        with pytest.raises(ValueError, match="attack_content cannot be empty"):
            workflow._validate_context(context=context)

    def test_validate_context_raises_error_for_missing_processing_callback(self, mock_attack_setup_target):
        """Test validation fails with missing processing callback."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
        )

        # Mock the callback to be falsy for testing validation
        with patch.object(context, "processing_callback", None):
            with pytest.raises(ValueError, match="processing_callback is required"):
                workflow._validate_context(context=context)

    def test_validate_context_with_additional_optional_fields(self, mock_attack_setup_target):
        """Test validation passes with additional optional fields."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
            processing_prompt="optional processing prompt",
            memory_labels={"key": "value"},
        )

        # Should not raise any exception
        workflow._validate_context(context=context)


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowSetupPhase:
    """Tests for the setup phase of the workflow."""

    @pytest.mark.asyncio
    async def test_setup_generates_new_conversation_ids(self, mock_attack_setup_target, basic_context):
        """Test that setup generates new conversation IDs."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        original_attack_id = basic_context.attack_setup_target_conversation_id

        await workflow._setup_async(context=basic_context)

        # Should generate a new conversation ID
        assert basic_context.attack_setup_target_conversation_id != original_attack_id

    @pytest.mark.asyncio
    async def test_setup_combines_memory_labels(self, mock_attack_setup_target, basic_context):
        """Test that setup combines workflow and context memory labels."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)
        workflow._memory_labels = {"workflow": "label"}

        basic_context.memory_labels = {"context": "label"}
        original_context_labels = basic_context.memory_labels.copy()

        with patch("pyrit.executor.workflow.xpia.combine_dict") as mock_combine:
            mock_combine.return_value = {"combined": "labels"}

            await workflow._setup_async(context=basic_context)

            mock_combine.assert_called_once_with(workflow._memory_labels, original_context_labels)
            assert basic_context.memory_labels == {"combined": "labels"}

    @pytest.mark.asyncio
    async def test_setup_with_empty_workflow_memory_labels(self, mock_attack_setup_target, basic_context):
        """Test setup when workflow has no memory labels."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)
        basic_context.memory_labels = {"existing": "label"}

        await workflow._setup_async(context=basic_context)

        # Should preserve existing labels
        assert "existing" in basic_context.memory_labels


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowAttackSetup:
    """Tests for the attack setup phase."""

    @pytest.mark.asyncio
    async def test_setup_attack_creates_seed_prompt_group(
        self, mock_attack_setup_target, mock_prompt_normalizer, basic_context, sample_attack_response
    ):
        """Test that setup attack creates proper seed prompt group."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            prompt_normalizer=mock_prompt_normalizer,
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response

        await workflow._setup_attack_async(context=basic_context)

        # Verify send_prompt_async was called with correct parameters
        call_args = mock_prompt_normalizer.send_prompt_async.call_args
        seed_prompt_group = call_args.kwargs["seed_prompt_group"]

        assert len(seed_prompt_group.prompts) == 1
        assert seed_prompt_group.prompts[0].value == basic_context.attack_content
        assert seed_prompt_group.prompts[0].data_type == "text"

    @pytest.mark.asyncio
    async def test_setup_attack_uses_correct_target_and_labels(
        self, mock_attack_setup_target, mock_prompt_normalizer, basic_context, sample_attack_response
    ):
        """Test that setup attack uses correct target and labels."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.memory_labels = {"test": "label"}
        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response

        await workflow._setup_attack_async(context=basic_context)

        # Verify send_prompt_async was called with correct parameters
        call_args = mock_prompt_normalizer.send_prompt_async.call_args

        assert call_args.kwargs["target"] == mock_attack_setup_target
        assert call_args.kwargs["labels"] == basic_context.memory_labels
        assert call_args.kwargs["conversation_id"] == basic_context.attack_setup_target_conversation_id

    @pytest.mark.asyncio
    async def test_setup_attack_uses_request_converters(
        self,
        mock_attack_setup_target,
        mock_prompt_normalizer,
        mock_converter_config,
        basic_context,
        sample_attack_response,
    ):
        """Test that setup attack uses configured request converters."""
        mock_converter_config.request_converters = [MagicMock()]

        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            converter_config=mock_converter_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response

        await workflow._setup_attack_async(context=basic_context)

        # Verify send_prompt_async was called with request converters
        call_args = mock_prompt_normalizer.send_prompt_async.call_args
        assert call_args.kwargs["request_converter_configurations"] == mock_converter_config.request_converters

    @pytest.mark.asyncio
    async def test_setup_attack_returns_response_text(
        self, mock_attack_setup_target, mock_prompt_normalizer, basic_context, sample_attack_response
    ):
        """Test that setup attack returns response text correctly."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            prompt_normalizer=mock_prompt_normalizer,
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response

        result = await workflow._setup_attack_async(context=basic_context)

        assert result == sample_attack_response.get_value()


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowProcessingExecution:
    """Tests for the processing execution phase."""

    @pytest.mark.asyncio
    async def test_execute_processing_calls_callback(self, mock_attack_setup_target, basic_context):
        """Test that execute processing calls the processing callback."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        basic_context.processing_callback = AsyncMock(return_value="callback response")

        result = await workflow._execute_processing_async(context=basic_context)

        basic_context.processing_callback.assert_called_once()
        assert result == "callback response"

    @pytest.mark.asyncio
    async def test_execute_processing_with_different_callback_responses(self, mock_attack_setup_target):
        """Test execute processing with various callback response types."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        test_responses = ["simple text", "complex response with multiple words", ""]

        for response in test_responses:
            context = XPIAContext(
                attack_content="test content",
                processing_callback=AsyncMock(return_value=response),
            )

            result = await workflow._execute_processing_async(context=context)
            assert result == response


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowScoringPhase:
    """Tests for the scoring phase."""

    @pytest.mark.asyncio
    async def test_score_response_with_no_scorer_returns_none(self, mock_attack_setup_target):
        """Test that scoring returns None when no scorer is configured."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=None,
        )

        result = await workflow._score_response_async(processing_response="test response")

        assert result is None

    @pytest.mark.asyncio
    async def test_score_response_with_scorer_returns_score(self, mock_attack_setup_target, mock_scorer, success_score):
        """Test that scoring returns score when scorer is configured."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
        )

        mock_scorer.score_text_async.return_value = [success_score]

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor
            mock_submit_result = MagicMock()
            mock_submit_result.result.return_value = [success_score]
            mock_executor.submit.return_value = mock_submit_result

            result = await workflow._score_response_async(processing_response="test response")

            assert result == success_score
            mock_executor.shutdown.assert_called_once_with(wait=False)

    @pytest.mark.asyncio
    async def test_score_response_handles_scorer_exception(self, mock_attack_setup_target, mock_scorer):
        """Test that scoring handles exceptions gracefully."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
        )

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor
            mock_submit_result = MagicMock()
            mock_submit_result.result.side_effect = Exception("Scoring error")
            mock_executor.submit.return_value = mock_submit_result

            result = await workflow._score_response_async(processing_response="test response")

            assert result is None
            mock_executor.shutdown.assert_called_once_with(wait=False)

    @pytest.mark.asyncio
    async def test_score_response_uses_thread_pool_executor(self, mock_attack_setup_target, mock_scorer, success_score):
        """Test that scoring uses ThreadPoolExecutor as expected."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
        )

        mock_scorer.score_text_async.return_value = [success_score]

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor
            mock_submit_result = MagicMock()
            mock_submit_result.result.return_value = [success_score]
            mock_executor.submit.return_value = mock_submit_result

            await workflow._score_response_async(processing_response="test response")

            # Verify ThreadPoolExecutor was used correctly
            mock_executor_class.assert_called_once()
            mock_executor.submit.assert_called_once()
            # Verify asyncio.run and scorer.score_text_async would be called
            submit_call_args = mock_executor.submit.call_args[0]
            assert submit_call_args[0] == asyncio.run


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowPerformPhase:
    """Tests for the perform phase that orchestrates the complete workflow."""

    @pytest.mark.asyncio
    async def test_perform_async_orchestrates_complete_workflow(
        self, mock_attack_setup_target, mock_prompt_normalizer, basic_context, sample_attack_response
    ):
        """Test that perform async orchestrates the complete workflow correctly."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            prompt_normalizer=mock_prompt_normalizer,
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response
        basic_context.processing_callback = AsyncMock(return_value="processing response")

        result = await workflow._perform_async(context=basic_context)

        assert isinstance(result, XPIAResult)
        assert result.processing_response == "processing response"
        assert result.attack_setup_response == sample_attack_response.get_value()
        assert result.score is None  # No scorer configured

    @pytest.mark.asyncio
    async def test_perform_async_with_scorer(
        self,
        mock_attack_setup_target,
        mock_prompt_normalizer,
        mock_scorer,
        basic_context,
        sample_attack_response,
        success_score,
    ):
        """Test perform async with scorer configured."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
            prompt_normalizer=mock_prompt_normalizer,
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response
        basic_context.processing_callback = AsyncMock(return_value="processing response")

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor
            mock_submit_result = MagicMock()
            mock_submit_result.result.return_value = [success_score]
            mock_executor.submit.return_value = mock_submit_result

            result = await workflow._perform_async(context=basic_context)

            assert isinstance(result, XPIAResult)
            assert result.processing_response == "processing response"
            assert result.attack_setup_response == sample_attack_response.get_value()
            assert result.score == success_score

    @pytest.mark.asyncio
    async def test_perform_async_calls_methods_in_correct_order(self, mock_attack_setup_target, basic_context):
        """Test that perform async calls methods in the correct order."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        # Mock all the internal methods
        with patch.object(workflow, "_setup_attack_async", new_callable=AsyncMock) as mock_setup:
            with patch.object(workflow, "_execute_processing_async", new_callable=AsyncMock) as mock_execute:
                with patch.object(workflow, "_score_response_async", new_callable=AsyncMock) as mock_score:

                    mock_setup.return_value = "setup response"
                    mock_execute.return_value = "processing response"
                    mock_score.return_value = None

                    await workflow._perform_async(context=basic_context)

                    # Verify methods were called in correct order
                    mock_setup.assert_called_once_with(context=basic_context)
                    mock_execute.assert_called_once_with(context=basic_context)
                    mock_score.assert_called_once_with(processing_response="processing response")


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowExecuteAsyncOverloads:
    """Tests for the execute_async method overloads."""

    @pytest.mark.asyncio
    async def test_execute_async_with_attack_content(self, mock_attack_setup_target):
        """Test execute_async with attack_content parameter."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        # Mock the parent execute_async method
        with patch.object(workflow.__class__.__bases__[0], "execute_async", new_callable=AsyncMock) as mock_parent:
            mock_result = XPIAResult(processing_response="test response")
            mock_parent.return_value = mock_result

            await workflow.execute_async(attack_content="test attack")

            # Verify parent execute_async was called with correct parameters
            mock_parent.assert_called_once()
            call_kwargs = mock_parent.call_args.kwargs
            assert call_kwargs["attack_content"] == "test attack"
            assert call_kwargs["processing_prompt"] == ""
            assert call_kwargs["memory_labels"] == {}

    @pytest.mark.asyncio
    async def test_execute_async_with_all_optional_parameters(self, mock_attack_setup_target):
        """Test execute_async with all optional parameters provided."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        with patch.object(workflow.__class__.__bases__[0], "execute_async", new_callable=AsyncMock) as mock_parent:
            mock_result = XPIAResult(processing_response="test response")
            mock_parent.return_value = mock_result

            memory_labels = {"test": "label"}
            await workflow.execute_async(
                attack_content="test attack",
                processing_prompt="test processing",
                memory_labels=memory_labels,
            )

            call_kwargs = mock_parent.call_args.kwargs
            assert call_kwargs["attack_content"] == "test attack"
            assert call_kwargs["processing_prompt"] == "test processing"
            assert call_kwargs["memory_labels"] == memory_labels

    @pytest.mark.asyncio
    async def test_execute_async_with_additional_kwargs(self, mock_attack_setup_target):
        """Test execute_async passes through additional kwargs."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        with patch.object(workflow.__class__.__bases__[0], "execute_async", new_callable=AsyncMock) as mock_parent:
            mock_result = XPIAResult(processing_response="test response")
            mock_parent.return_value = mock_result

            await workflow.execute_async(
                attack_content="test attack",
                custom_param="custom_value",
            )

            call_kwargs = mock_parent.call_args.kwargs
            assert "custom_param" in call_kwargs
            assert call_kwargs["custom_param"] == "custom_value"


@pytest.mark.usefixtures("patch_central_database")
class TestXPIATestWorkflowContextValidation:
    """Tests for XPIATestWorkflow context validation."""

    def test_validate_context_with_valid_context(self, mock_attack_setup_target, mock_processing_target, mock_scorer):
        """Test that valid context passes validation without errors."""
        workflow = XPIATestWorkflow(
            attack_setup_target=mock_attack_setup_target,
            processing_target=mock_processing_target,
            scorer=mock_scorer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
            processing_prompt="test processing prompt",
        )

        # Should not raise any exception
        workflow._validate_context(context=context)

    def test_validate_context_raises_error_for_empty_processing_prompt(
        self, mock_attack_setup_target, mock_processing_target, mock_scorer
    ):
        """Test validation fails with empty processing prompt."""
        workflow = XPIATestWorkflow(
            attack_setup_target=mock_attack_setup_target,
            processing_target=mock_processing_target,
            scorer=mock_scorer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
            processing_prompt="",  # Empty processing prompt
        )

        with pytest.raises(ValueError, match="processing_prompt cannot be empty"):
            workflow._validate_context(context=context)

    def test_validate_context_calls_parent_validation(
        self, mock_attack_setup_target, mock_processing_target, mock_scorer
    ):
        """Test that validate_context calls parent validation."""
        workflow = XPIATestWorkflow(
            attack_setup_target=mock_attack_setup_target,
            processing_target=mock_processing_target,
            scorer=mock_scorer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
            processing_prompt="test processing prompt",
        )

        # Mock the parent _validate_context method
        with patch.object(workflow.__class__.__bases__[0], "_validate_context") as mock_parent_validate:
            workflow._validate_context(context=context)

            # Verify parent validation was called
            mock_parent_validate.assert_called_once_with(context=context)


@pytest.mark.usefixtures("patch_central_database")
class TestXPIATestWorkflowSetupPhase:
    """Tests for XPIATestWorkflow setup phase."""

    @pytest.mark.asyncio
    async def test_setup_creates_processing_callback(
        self,
        mock_attack_setup_target,
        mock_processing_target,
        mock_scorer,
        mock_prompt_normalizer,
        sample_processing_response,
    ):
        """Test that setup creates the processing callback correctly."""
        workflow = XPIATestWorkflow(
            attack_setup_target=mock_attack_setup_target,
            processing_target=mock_processing_target,
            scorer=mock_scorer,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
            processing_prompt="test processing prompt",
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_processing_response

        await workflow._setup_async(context=context)

        # Verify that processing_callback was set
        assert context.processing_callback is not None

        # Test the callback functionality
        result = await context.processing_callback()
        assert result == sample_processing_response.get_value()

    @pytest.mark.asyncio
    async def test_setup_callback_uses_correct_parameters(
        self,
        mock_attack_setup_target,
        mock_processing_target,
        mock_scorer,
        mock_prompt_normalizer,
        sample_processing_response,
    ):
        """Test that the processing callback uses correct parameters."""
        workflow = XPIATestWorkflow(
            attack_setup_target=mock_attack_setup_target,
            processing_target=mock_processing_target,
            scorer=mock_scorer,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
            processing_prompt="test processing prompt",
            memory_labels={"test": "label"},
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_processing_response

        await workflow._setup_async(context=context)

        # Execute the callback to verify it uses correct parameters
        await context.processing_callback()

        # Verify send_prompt_async was called with correct parameters
        call_args = mock_prompt_normalizer.send_prompt_async.call_args
        seed_prompt_group = call_args.kwargs["seed_prompt_group"]

        assert len(seed_prompt_group.prompts) == 1
        assert seed_prompt_group.prompts[0].value == "test processing prompt"
        assert seed_prompt_group.prompts[0].data_type == "text"
        assert call_args.kwargs["target"] == mock_processing_target
        assert call_args.kwargs["labels"] == context.memory_labels
        assert call_args.kwargs["conversation_id"] == context.processing_conversation_id

    @pytest.mark.asyncio
    async def test_setup_calls_parent_setup(self, mock_attack_setup_target, mock_processing_target, mock_scorer):
        """Test that setup calls parent setup method."""
        workflow = XPIATestWorkflow(
            attack_setup_target=mock_attack_setup_target,
            processing_target=mock_processing_target,
            scorer=mock_scorer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
            processing_prompt="test processing prompt",
        )

        # Mock the parent _setup_async method
        with patch.object(workflow.__class__.__bases__[0], "_setup_async", new_callable=AsyncMock) as mock_parent:
            await workflow._setup_async(context=context)

            # Verify parent setup was called
            mock_parent.assert_called_once_with(context=context)


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAManualProcessingWorkflowContextValidation:
    """Tests for XPIAManualProcessingWorkflow context validation."""

    def test_validate_context_with_valid_context(self, mock_attack_setup_target, mock_scorer):
        """Test that valid context passes validation without errors."""
        workflow = XPIAManualProcessingWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
        )

        # Should not raise any exception (processing_callback will be overridden)
        workflow._validate_context(context=context)

    def test_validate_context_raises_error_for_empty_attack_content(self, mock_attack_setup_target, mock_scorer):
        """Test validation fails with empty attack content."""
        workflow = XPIAManualProcessingWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
        )

        context = XPIAContext(
            attack_content="",  # Empty attack content
            processing_callback=AsyncMock(return_value="test response"),
        )

        with pytest.raises(ValueError, match="attack_content cannot be empty"):
            workflow._validate_context(context=context)

    def test_validate_context_accepts_missing_processing_callback(self, mock_attack_setup_target, mock_scorer):
        """Test validation passes when processing callback is missing (will be set by setup)."""
        workflow = XPIAManualProcessingWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
        )

        # Should not raise any exception (manual workflow will override the callback)
        workflow._validate_context(context=context)


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAManualProcessingWorkflowSetupPhase:
    """Tests for XPIAManualProcessingWorkflow setup phase."""

    @pytest.mark.asyncio
    async def test_setup_creates_manual_input_callback(self, mock_attack_setup_target, mock_scorer):
        """Test that setup creates the manual input callback correctly."""
        workflow = XPIAManualProcessingWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
        )

        with patch("pyrit.executor.workflow.xpia.ainput", new_callable=AsyncMock) as mock_ainput:
            mock_ainput.return_value = "Manual input response"

            await workflow._setup_async(context=context)

            # Verify that processing_callback was set
            assert context.processing_callback is not None

            # Test the callback functionality
            result = await context.processing_callback()
            assert result == "Manual input response"

            # Verify ainput was called with correct prompt
            mock_ainput.assert_called_once_with(
                "Please trigger the processing target's execution and paste the output here: "
            )

    @pytest.mark.asyncio
    async def test_setup_calls_parent_setup(self, mock_attack_setup_target, mock_scorer):
        """Test that setup calls parent setup method."""
        workflow = XPIAManualProcessingWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
        )

        # Mock the parent _setup_async method
        with patch.object(workflow.__class__.__bases__[0], "_setup_async", new_callable=AsyncMock) as mock_parent:
            with patch("pyrit.executor.workflow.xpia.ainput", new_callable=AsyncMock):
                await workflow._setup_async(context=context)

                # Verify parent setup was called
                mock_parent.assert_called_once_with(context=context)

    @pytest.mark.asyncio
    async def test_manual_input_callback_functionality(self, mock_attack_setup_target, mock_scorer):
        """Test the manual input callback works correctly with different responses."""
        workflow = XPIAManualProcessingWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),
        )

        test_inputs = ["Simple response", "Complex multi-line\nresponse", ""]

        for test_input in test_inputs:
            with patch("pyrit.executor.workflow.xpia.ainput", new_callable=AsyncMock) as mock_ainput:
                mock_ainput.return_value = test_input

                await workflow._setup_async(context=context)
                result = await context.processing_callback()

                assert result == test_input


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowIntegration:
    """Integration tests for complete workflow functionality."""

    @pytest.mark.asyncio
    async def test_complete_workflow_execution_without_scorer(
        self, mock_attack_setup_target, mock_prompt_normalizer, sample_attack_response
    ):
        """Test complete workflow execution without scorer."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="processing response"),
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response

        # Mock the individual workflow steps
        with patch.object(workflow, "_validate_context") as mock_validate:
            with patch.object(workflow, "_setup_async", new_callable=AsyncMock) as mock_setup:
                with patch.object(workflow, "_teardown_async", new_callable=AsyncMock) as mock_teardown:

                    result = await workflow.execute_with_context_async(context=context)

                    # Verify all phases were called
                    mock_validate.assert_called_once_with(context=context)
                    mock_setup.assert_called_once_with(context=context)
                    mock_teardown.assert_called_once_with(context=context)

                    # Verify result
                    assert isinstance(result, XPIAResult)
                    assert result.processing_response == "processing response"
                    assert result.score is None

    @pytest.mark.asyncio
    async def test_complete_workflow_execution_with_scorer(
        self, mock_attack_setup_target, mock_prompt_normalizer, mock_scorer, sample_attack_response, success_score
    ):
        """Test complete workflow execution with scorer."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="processing response"),
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor
            mock_submit_result = MagicMock()
            mock_submit_result.result.return_value = [success_score]
            mock_executor.submit.return_value = mock_submit_result

            result = await workflow.execute_with_context_async(context=context)

            # Verify result includes score
            assert isinstance(result, XPIAResult)
            assert result.processing_response == "processing response"
            assert result.score == success_score
            assert result.attack_setup_response == sample_attack_response.get_value()

    @pytest.mark.asyncio
    async def test_test_workflow_integration(
        self,
        mock_attack_setup_target,
        mock_processing_target,
        mock_scorer,
        mock_prompt_normalizer,
        sample_attack_response,
        sample_processing_response,
        success_score,
    ):
        """Test complete XPIATestWorkflow integration."""
        workflow = XPIATestWorkflow(
            attack_setup_target=mock_attack_setup_target,
            processing_target=mock_processing_target,
            scorer=mock_scorer,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),  # Will be overridden
            processing_prompt="test processing prompt",
        )

        mock_prompt_normalizer.send_prompt_async.side_effect = [sample_attack_response, sample_processing_response]

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor
            mock_submit_result = MagicMock()
            mock_submit_result.result.return_value = [success_score]
            mock_executor.submit.return_value = mock_submit_result

            result = await workflow.execute_with_context_async(context=context)

            # Verify result
            assert isinstance(result, XPIAResult)
            assert result.processing_response == sample_processing_response.get_value()
            assert result.score == success_score
            assert result.attack_setup_response == sample_attack_response.get_value()

            # Verify both targets were used
            assert mock_prompt_normalizer.send_prompt_async.call_count == 2

    @pytest.mark.asyncio
    async def test_manual_workflow_integration(
        self, mock_attack_setup_target, mock_scorer, mock_prompt_normalizer, sample_attack_response, success_score
    ):
        """Test complete XPIAManualProcessingWorkflow integration."""
        workflow = XPIAManualProcessingWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="test response"),  # Will be overridden
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response

        with patch("pyrit.executor.workflow.xpia.ainput", new_callable=AsyncMock) as mock_ainput:
            mock_ainput.return_value = "Manual processing response"

            with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
                mock_executor = MagicMock()
                mock_executor_class.return_value = mock_executor
                mock_submit_result = MagicMock()
                mock_submit_result.result.return_value = [success_score]
                mock_executor.submit.return_value = mock_submit_result

                result = await workflow.execute_with_context_async(context=context)

                # Verify result
                assert isinstance(result, XPIAResult)
                assert result.processing_response == "Manual processing response"
                assert result.score == success_score
                assert result.attack_setup_response == sample_attack_response.get_value()

                # Verify manual input was requested
                mock_ainput.assert_called_once()


@pytest.mark.usefixtures("patch_central_database")
class TestXPIAWorkflowErrorHandling:
    """Tests for error handling and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_workflow_handles_attack_setup_failure(self, mock_attack_setup_target, mock_prompt_normalizer):
        """Test workflow handles attack setup failure gracefully."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="processing response"),
        )

        mock_prompt_normalizer.send_prompt_async.side_effect = Exception("Attack setup failed")

        with pytest.raises(Exception, match="Attack setup failed"):
            await workflow.execute_with_context_async(context=context)

    @pytest.mark.asyncio
    async def test_workflow_handles_processing_callback_failure(
        self, mock_attack_setup_target, mock_prompt_normalizer, sample_attack_response
    ):
        """Test workflow handles processing callback failure gracefully."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(side_effect=Exception("Processing failed")),
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response

        with pytest.raises(Exception, match="Processing failed"):
            await workflow.execute_with_context_async(context=context)

    @pytest.mark.asyncio
    async def test_workflow_handles_validation_errors(self, mock_attack_setup_target):
        """Test workflow handles context validation errors."""
        workflow = XPIAWorkflow(attack_setup_target=mock_attack_setup_target)

        # Context with missing attack content
        context = XPIAContext(
            attack_content="",  # Invalid empty content
            processing_callback=AsyncMock(return_value="processing response"),
        )

        with pytest.raises(ValueError, match="attack_content cannot be empty"):
            await workflow.execute_with_context_async(context=context)

    @pytest.mark.asyncio
    async def test_scorer_error_handling_returns_none_score(
        self, mock_attack_setup_target, mock_scorer, mock_prompt_normalizer, sample_attack_response
    ):
        """Test that scorer errors are handled and return None score."""
        workflow = XPIAWorkflow(
            attack_setup_target=mock_attack_setup_target,
            scorer=mock_scorer,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = XPIAContext(
            attack_content="test attack content",
            processing_callback=AsyncMock(return_value="processing response"),
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_attack_response

        # Mock scorer to raise an exception
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor
            mock_submit_result = MagicMock()
            mock_submit_result.result.side_effect = Exception("Scoring failed")
            mock_executor.submit.return_value = mock_submit_result

            result = await workflow.execute_with_context_async(context=context)

            # Should still return result with None score
            assert isinstance(result, XPIAResult)
            assert result.processing_response == "processing response"
            assert result.score is None  # Error should result in None score
