# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from pyrit.attacks.base.attack_config import AttackConverterConfig, AttackScoringConfig
from pyrit.attacks.single_turn.anecdoctor import (
    AnecdoctorAttack,
    AnecdoctorAttackContext,
)
from pyrit.models import AttackOutcome, AttackResult, PromptRequestResponse, Score
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget


@pytest.fixture
def mock_objective_target() -> PromptChatTarget:
    """Create a mock objective target for testing."""
    mock_target = MagicMock(spec=PromptChatTarget)
    mock_target.set_system_prompt = MagicMock()
    return mock_target


@pytest.fixture
def mock_processing_model() -> PromptChatTarget:
    """Create a mock processing model for testing."""
    mock_model = MagicMock(spec=PromptChatTarget)
    mock_model.set_system_prompt = MagicMock()
    return mock_model


@pytest.fixture
def mock_prompt_normalizer() -> PromptNormalizer:
    """Create a mock prompt normalizer for testing."""
    mock_normalizer = MagicMock(spec=PromptNormalizer)
    mock_normalizer.send_prompt_async = AsyncMock()
    return mock_normalizer


@pytest.fixture
def sample_evaluation_data() -> List[str]:
    """Sample evaluation data for testing."""
    return [
        "Claim: The earth is flat. Review: FALSE",
        "Claim: Vaccines cause autism. Review: FALSE",
        "Claim: COVID-19 was engineered in a lab. Review: DISPUTED",
    ]


@pytest.fixture
def sample_context(sample_evaluation_data) -> AnecdoctorAttackContext:
    """Create a sample context for testing."""
    return AnecdoctorAttackContext(
        objective="Generate misleading content",
        memory_labels={"test": "label"},
        evaluation_data=sample_evaluation_data,
        language="english",
        content_type="viral tweet",
    )


@pytest.fixture
def mock_response() -> PromptRequestResponse:
    """Create a mock response for testing."""
    mock_response = MagicMock(spec=PromptRequestResponse)
    mock_response.get_piece.return_value = "Generated misinformation content"
    mock_response.get_value.return_value = "Generated misinformation content"
    return mock_response


@pytest.fixture
def mock_score() -> Score:
    """Create a mock score for testing."""
    mock_score = MagicMock(spec=Score)
    mock_score.get_value.return_value = True
    return mock_score


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackContext:
    """Tests for AnecdoctorAttackContext initialization and validation."""

    def test_context_initialization_with_defaults(self):
        """Test context initialization with default values."""
        context = AnecdoctorAttackContext(objective="test objective", memory_labels={"test": "label"})

        assert context.objective == "test objective"
        assert context.memory_labels == {"test": "label"}
        assert context.evaluation_data == []
        assert context.language == "english"
        assert context.content_type == "viral tweet"
        assert isinstance(context.conversation_id, str)

    def test_context_initialization_with_custom_values(self, sample_evaluation_data):
        """Test context initialization with custom values."""
        custom_id = str(uuid.uuid4())
        context = AnecdoctorAttackContext(
            objective="custom objective",
            memory_labels={"custom": "label"},
            evaluation_data=sample_evaluation_data,
            language="german",
            content_type="news article",
            conversation_id=custom_id,
        )

        assert context.objective == "custom objective"
        assert context.memory_labels == {"custom": "label"}
        assert context.evaluation_data == sample_evaluation_data
        assert context.language == "german"
        assert context.content_type == "news article"
        assert context.conversation_id == custom_id

    def test_create_from_params_success(self, sample_evaluation_data):
        """Test successful creation from parameters."""
        context = AnecdoctorAttackContext.create_from_params(
            objective="test objective",
            prepended_conversation=[],
            memory_labels={"test": "label"},
            evaluation_data=sample_evaluation_data,
            language="french",
            content_type="blog post",
        )

        assert context.objective == "test objective"
        assert context.memory_labels == {"test": "label"}
        assert context.evaluation_data == sample_evaluation_data
        assert context.language == "french"
        assert context.content_type == "blog post"

    def test_create_from_params_with_defaults(self):
        """Test creation from parameters with default values."""
        evaluation_data = ["test claim"]
        context = AnecdoctorAttackContext.create_from_params(
            objective="test objective",
            prepended_conversation=[],
            memory_labels={"test": "label"},
            evaluation_data=evaluation_data,
        )

        assert context.language == "english"
        assert context.content_type == "viral tweet"

    def test_create_from_params_empty_evaluation_data(self):
        """Test creation fails with empty evaluation data."""
        with pytest.raises(ValueError, match="evaluation_data cannot be empty"):
            AnecdoctorAttackContext.create_from_params(
                objective="test objective",
                prepended_conversation=[],
                memory_labels={"test": "label"},
                evaluation_data=[],
            )

    def test_create_from_params_invalid_evaluation_data_type(self):
        """Test creation fails with invalid evaluation data type."""
        with pytest.raises(ValueError, match="evaluation_data must be a list, got str"):
            AnecdoctorAttackContext.create_from_params(
                objective="test objective",
                prepended_conversation=[],
                memory_labels={"test": "label"},
                evaluation_data="not a list",
            )

    def test_create_from_params_invalid_language_type(self, sample_evaluation_data):
        """Test creation fails with invalid language type."""
        with pytest.raises(ValueError, match="language must be a string, got int"):
            AnecdoctorAttackContext.create_from_params(
                objective="test objective",
                prepended_conversation=[],
                memory_labels={"test": "label"},
                evaluation_data=sample_evaluation_data,
                language=123,
            )

    def test_create_from_params_invalid_content_type(self, sample_evaluation_data):
        """Test creation fails with invalid content type."""
        with pytest.raises(ValueError, match="content_type must be a string, got list"):
            AnecdoctorAttackContext.create_from_params(
                objective="test objective",
                prepended_conversation=[],
                memory_labels={"test": "label"},
                evaluation_data=sample_evaluation_data,
                content_type=["not", "a", "string"],
            )


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackInitialization:
    """Tests for AnecdoctorAttack initialization."""

    def test_init_minimal_parameters(self, mock_objective_target):
        """Test initialization with minimal required parameters."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        assert attack._objective_target == mock_objective_target
        assert attack._processing_model is None
        assert attack._request_converters == []
        assert attack._response_converters == []
        assert attack._auxiliary_scorers == []
        assert attack._objective_scorer is None
        assert isinstance(attack._prompt_normalizer, PromptNormalizer)

    def test_init_with_processing_model(self, mock_objective_target, mock_processing_model):
        """Test initialization with processing model."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target, processing_model=mock_processing_model)

        assert attack._objective_target == mock_objective_target
        assert attack._processing_model == mock_processing_model

    def test_init_with_custom_normalizer(self, mock_objective_target, mock_prompt_normalizer):
        """Test initialization with custom prompt normalizer."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target, prompt_normalizer=mock_prompt_normalizer)

        assert attack._prompt_normalizer == mock_prompt_normalizer

    def test_init_with_converter_config(self, mock_objective_target):
        """Test initialization with converter configuration."""
        mock_converter = MagicMock()
        converter_config = AttackConverterConfig(
            request_converters=[mock_converter], response_converters=[mock_converter]
        )

        attack = AnecdoctorAttack(objective_target=mock_objective_target, attack_converter_config=converter_config)

        assert attack._request_converters == [mock_converter]
        assert attack._response_converters == [mock_converter]

    def test_init_with_scoring_config(self, mock_objective_target):
        """Test initialization with scoring configuration."""
        mock_auxiliary_scorer = MagicMock()
        mock_objective_scorer = MagicMock()
        mock_objective_scorer.scorer_type = "true_false"

        scoring_config = AttackScoringConfig(
            auxiliary_scorers=[mock_auxiliary_scorer], objective_scorer=mock_objective_scorer
        )

        attack = AnecdoctorAttack(objective_target=mock_objective_target, attack_scoring_config=scoring_config)

        assert attack._auxiliary_scorers == [mock_auxiliary_scorer]
        assert attack._objective_scorer == mock_objective_scorer

    def test_init_invalid_objective_scorer_type(self, mock_objective_target):
        """Test initialization fails with invalid objective scorer type."""
        mock_objective_scorer = MagicMock()
        mock_objective_scorer.scorer_type = "invalid_type"

        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        with pytest.raises(ValueError, match="Objective scorer must be a true/false scorer"):
            AnecdoctorAttack(objective_target=mock_objective_target, attack_scoring_config=scoring_config)


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackValidation:
    """Tests for context validation."""

    def test_validate_context_success(self, mock_objective_target, sample_context):
        """Test successful context validation."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        # Should not raise any exception
        attack._validate_context(context=sample_context)

    def test_validate_context_empty_content_type(self, mock_objective_target, sample_context):
        """Test validation fails with empty content type."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        sample_context.content_type = ""

        with pytest.raises(ValueError, match="content_type must be provided in the context"):
            attack._validate_context(context=sample_context)

    def test_validate_context_empty_language(self, mock_objective_target, sample_context):
        """Test validation fails with empty language."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        sample_context.language = ""

        with pytest.raises(ValueError, match="language must be provided in the context"):
            attack._validate_context(context=sample_context)

    def test_validate_context_empty_evaluation_data(self, mock_objective_target, sample_context):
        """Test validation fails with empty evaluation data."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        sample_context.evaluation_data = []

        with pytest.raises(ValueError, match="evaluation_data cannot be empty"):
            attack._validate_context(context=sample_context)


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackSetup:
    """Tests for attack setup."""

    @pytest.mark.asyncio
    async def test_setup_without_processing_model(self, mock_objective_target, sample_context):
        """Test setup without processing model uses few-shot template."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.return_value = "System prompt for {language} {type}"

            await attack._setup_async(context=sample_context)

            mock_load.assert_called_once_with(yaml_filename="anecdoctor_use_fewshot.yaml")
            mock_objective_target.set_system_prompt.assert_called_once()

            # Verify the system prompt was formatted correctly
            call_args = mock_objective_target.set_system_prompt.call_args
            assert "System prompt for english viral tweet" in call_args.kwargs["system_prompt"]

    @pytest.mark.asyncio
    async def test_setup_with_processing_model(self, mock_objective_target, mock_processing_model, sample_context):
        """Test setup with processing model uses knowledge graph template."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target, processing_model=mock_processing_model)

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.return_value = "KG prompt for {language} {type}"

            await attack._setup_async(context=sample_context)

            mock_load.assert_called_once_with(yaml_filename="anecdoctor_use_knowledge_graph.yaml")
            mock_objective_target.set_system_prompt.assert_called_once()

            # Verify the system prompt was formatted correctly
            call_args = mock_objective_target.set_system_prompt.call_args
            assert "KG prompt for english viral tweet" in call_args.kwargs["system_prompt"]

    @pytest.mark.asyncio
    async def test_setup_generates_conversation_id(self, mock_objective_target, sample_context):
        """Test setup generates a new conversation ID."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        original_id = sample_context.conversation_id

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.return_value = "Test prompt"

            await attack._setup_async(context=sample_context)

            # Conversation ID should be updated
            assert sample_context.conversation_id != original_id
            assert isinstance(sample_context.conversation_id, str)


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackExecution:
    """Tests for the main attack execution flow."""

    @pytest.mark.asyncio
    async def test_perform_attack_without_processing_model(self, mock_objective_target, sample_context, mock_response):
        """Test attack execution without processing model (few-shot mode)."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        with (
            patch.object(attack, "_prepare_examples_async") as mock_prepare,
            patch.object(attack, "_send_examples_to_target_async") as mock_send,
            patch.object(attack, "_evaluate_response_if_configured_async") as mock_eval,
        ):

            mock_prepare.return_value = "### examples\nFormatted examples"
            mock_send.return_value = mock_response
            mock_eval.return_value = None

            result = await attack._perform_attack_async(context=sample_context)

            # Verify method calls
            mock_prepare.assert_called_once_with(context=sample_context)
            mock_send.assert_called_once_with(
                formatted_examples="### examples\nFormatted examples", context=sample_context
            )
            mock_eval.assert_called_once_with(response=mock_response, objective=sample_context.objective)

            # Verify result
            assert isinstance(result, AttackResult)
            assert result.conversation_id == sample_context.conversation_id
            assert result.objective == sample_context.objective
            assert result.last_response == mock_response.get_piece()
            assert result.outcome == AttackOutcome.SUCCESS
            assert result.executed_turns == 1

    @pytest.mark.asyncio
    async def test_perform_attack_with_processing_model(
        self, mock_objective_target, mock_processing_model, sample_context, mock_response
    ):
        """Test attack execution with processing model (knowledge graph mode)."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target, processing_model=mock_processing_model)

        with (
            patch.object(attack, "_extract_knowledge_graph_async") as mock_kg,
            patch.object(attack, "_send_examples_to_target_async") as mock_send,
            patch.object(attack, "_evaluate_response_if_configured_async") as mock_eval,
        ):

            mock_kg.return_value = "Extracted knowledge graph"
            mock_send.return_value = mock_response
            mock_eval.return_value = None

            result = await attack._perform_attack_async(context=sample_context)

            # Verify knowledge graph extraction was called
            mock_kg.assert_called_once_with(context=sample_context)
            mock_send.assert_called_once_with(formatted_examples="Extracted knowledge graph", context=sample_context)

            assert result.outcome == AttackOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_perform_attack_with_scoring(self, mock_objective_target, sample_context, mock_response, mock_score):
        """Test attack execution with scoring configured."""
        mock_scorer = MagicMock()
        mock_scorer.scorer_type = "true_false"
        scoring_config = AttackScoringConfig(objective_scorer=mock_scorer)

        attack = AnecdoctorAttack(objective_target=mock_objective_target, attack_scoring_config=scoring_config)

        with (
            patch.object(attack, "_prepare_examples_async") as mock_prepare,
            patch.object(attack, "_send_examples_to_target_async") as mock_send,
            patch.object(attack, "_evaluate_response_async") as mock_eval,
        ):

            mock_prepare.return_value = "Formatted examples"
            mock_send.return_value = mock_response
            mock_eval.return_value = mock_score

            result = await attack._perform_attack_async(context=sample_context)

            mock_eval.assert_called_once_with(response=mock_response, objective=sample_context.objective)

            assert result.last_score == mock_score
            assert result.outcome == AttackOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_perform_attack_no_response(self, mock_objective_target, sample_context):
        """Test attack execution when no response is received."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        with (
            patch.object(attack, "_prepare_examples_async") as mock_prepare,
            patch.object(attack, "_send_examples_to_target_async") as mock_send,
        ):

            mock_prepare.return_value = "Formatted examples"
            mock_send.return_value = None

            result = await attack._perform_attack_async(context=sample_context)

            assert result.outcome == AttackOutcome.FAILURE
            assert result.outcome_reason is not None
            assert "no response received" in result.outcome_reason
            assert result.last_response is None


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackHelperMethods:
    """Tests for helper methods."""

    def test_format_few_shot_examples(self, mock_objective_target, sample_evaluation_data):
        """Test formatting of few-shot examples."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        result = attack._format_few_shot_examples(evaluation_data=sample_evaluation_data)

        expected = "### examples\n" + "\n".join(sample_evaluation_data)
        assert result == expected

    def test_load_prompt_from_yaml(self, mock_objective_target):
        """Test loading prompt from YAML file."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        mock_yaml_content = {"value": "Test prompt template"}

        with patch("pathlib.Path.read_text") as mock_read, patch("yaml.safe_load") as mock_yaml_load:

            mock_read.return_value = "yaml content"
            mock_yaml_load.return_value = mock_yaml_content

            result = attack._load_prompt_from_yaml(yaml_filename="test.yaml")

            assert result == "Test prompt template"
            mock_read.assert_called_once_with(encoding="utf-8")
            mock_yaml_load.assert_called_once_with("yaml content")

    @pytest.mark.asyncio
    async def test_extract_knowledge_graph_success(self, mock_objective_target, mock_processing_model, sample_context):
        """Test successful knowledge graph extraction."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target, processing_model=mock_processing_model)

        # Mock the normalizer and response
        mock_kg_response = MagicMock()
        mock_kg_response.get_value.return_value = "Extracted knowledge graph"
        attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=mock_kg_response)

        with (
            patch.object(attack, "_load_prompt_from_yaml") as mock_load,
            patch.object(attack, "_format_few_shot_examples") as mock_format,
        ):

            mock_load.return_value = "KG prompt for {language}"
            mock_format.return_value = "formatted examples"

            result = await attack._extract_knowledge_graph_async(context=sample_context)

            assert result == "Extracted knowledge graph"

            # Verify system prompt was set on processing model
            mock_processing_model.set_system_prompt.assert_called_once()

            # Verify prompt was sent to processing model
            attack._prompt_normalizer.send_prompt_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_knowledge_graph_no_response(
        self, mock_objective_target, mock_processing_model, sample_context
    ):
        """Test knowledge graph extraction when no response is received."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target, processing_model=mock_processing_model)

        attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=None)

        with (
            patch.object(attack, "_load_prompt_from_yaml") as mock_load,
            patch.object(attack, "_format_few_shot_examples") as mock_format,
        ):

            mock_load.return_value = "KG prompt for {language}"
            mock_format.return_value = "formatted examples"

            with pytest.raises(RuntimeError, match="Failed to extract knowledge graph"):
                await attack._extract_knowledge_graph_async(context=sample_context)

    @pytest.mark.asyncio
    async def test_send_examples_to_target(self, mock_objective_target, sample_context, mock_response):
        """Test sending examples to target model."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=mock_response)

        result = await attack._send_examples_to_target_async(formatted_examples="test examples", context=sample_context)

        assert result == mock_response

        # Verify prompt normalizer was called with correct parameters
        call_args = attack._prompt_normalizer.send_prompt_async.call_args
        assert call_args.kwargs["target"] == mock_objective_target
        assert call_args.kwargs["conversation_id"] == sample_context.conversation_id

    def test_determine_attack_outcome_no_scorer_success(self, mock_objective_target, mock_response):
        """Test outcome determination without scorer when response exists."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        outcome, reason = attack._determine_attack_outcome(response=mock_response, score=None)

        assert outcome == AttackOutcome.SUCCESS
        assert reason is not None
        assert "Successfully generated content" in reason

    def test_determine_attack_outcome_no_scorer_failure(self, mock_objective_target):
        """Test outcome determination without scorer when no response."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        outcome, reason = attack._determine_attack_outcome(response=None, score=None)

        assert outcome == AttackOutcome.FAILURE
        assert reason is not None
        assert "no response received" in reason

    def test_determine_attack_outcome_with_scorer_success(self, mock_objective_target, mock_response, mock_score):
        """Test outcome determination with scorer when objective achieved."""
        mock_scorer = MagicMock()
        mock_scorer.scorer_type = "true_false"
        scoring_config = AttackScoringConfig(objective_scorer=mock_scorer)

        attack = AnecdoctorAttack(objective_target=mock_objective_target, attack_scoring_config=scoring_config)

        outcome, reason = attack._determine_attack_outcome(response=mock_response, score=mock_score)

        assert outcome == AttackOutcome.SUCCESS
        assert reason is not None
        assert "Objective achieved according to scorer" in reason

    def test_determine_attack_outcome_with_scorer_failure(self, mock_objective_target, mock_response):
        """Test outcome determination with scorer when objective not achieved."""
        mock_scorer = MagicMock()
        mock_scorer.scorer_type = "true_false"
        scoring_config = AttackScoringConfig(objective_scorer=mock_scorer)

        attack = AnecdoctorAttack(objective_target=mock_objective_target, attack_scoring_config=scoring_config)

        mock_score = MagicMock()
        mock_score.get_value.return_value = False

        outcome, reason = attack._determine_attack_outcome(response=mock_response, score=mock_score)

        assert outcome == AttackOutcome.FAILURE
        assert reason is not None
        assert "objective not achieved according to scorer" in reason


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackIntegration:
    """Integration tests for complete attack execution."""

    @pytest.mark.asyncio
    async def test_complete_attack_flow_few_shot(self, mock_objective_target, sample_evaluation_data, mock_response):
        """Test complete attack flow in few-shot mode."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=mock_response)

        context = AnecdoctorAttackContext(
            objective="Generate misleading content",
            memory_labels={"test": "integration"},
            evaluation_data=sample_evaluation_data,
            language="spanish",
            content_type="news article",
        )

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.return_value = "System prompt for {language} {type}"

            # Setup
            await attack._setup_async(context=context)

            # Execute attack
            result = await attack._perform_attack_async(context=context)

            # Verify setup was called correctly
            mock_objective_target.set_system_prompt.assert_called_once()
            system_prompt_call = mock_objective_target.set_system_prompt.call_args
            assert "spanish" in system_prompt_call.kwargs["system_prompt"]
            assert "news article" in system_prompt_call.kwargs["system_prompt"]

            # Verify attack execution
            assert result.outcome == AttackOutcome.SUCCESS
            assert result.conversation_id == context.conversation_id
            assert result.objective == context.objective

    @pytest.mark.asyncio
    async def test_complete_attack_flow_with_kg(
        self, mock_objective_target, mock_processing_model, sample_evaluation_data, mock_response
    ):
        """Test complete attack flow with knowledge graph extraction."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target, processing_model=mock_processing_model)

        # Mock both target and processing model responses
        mock_kg_response = MagicMock()
        mock_kg_response.get_value.return_value = "Extracted KG"

        def mock_send_async(*args, **kwargs):
            # Return KG response for processing model, regular response for target
            if kwargs["target"] == mock_processing_model:
                return mock_kg_response
            return mock_response

        attack._prompt_normalizer.send_prompt_async = AsyncMock(side_effect=mock_send_async)

        context = AnecdoctorAttackContext(
            objective="Generate misleading content",
            memory_labels={"test": "integration"},
            evaluation_data=sample_evaluation_data,
            language="french",
            content_type="blog post",
        )

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.side_effect = lambda yaml_filename: f"Prompt template from {yaml_filename}"

            # Setup
            await attack._setup_async(context=context)

            # Execute attack
            result = await attack._perform_attack_async(context=context)

            # Verify both models had system prompts set
            mock_objective_target.set_system_prompt.assert_called_once()
            mock_processing_model.set_system_prompt.assert_called_once()

            # Verify attack execution
            assert result.outcome == AttackOutcome.SUCCESS
            assert result.conversation_id == context.conversation_id

    @pytest.mark.asyncio
    async def test_teardown_does_nothing(self, mock_objective_target, sample_context):
        """Test that teardown method completes without error."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        # Should complete without raising any exceptions
        await attack._teardown_async(context=sample_context)


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_yaml_file_not_found(self, mock_objective_target):
        """Test handling of missing YAML files."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = FileNotFoundError("File not found")

            with pytest.raises(FileNotFoundError):
                attack._load_prompt_from_yaml(yaml_filename="nonexistent.yaml")

    @pytest.mark.asyncio
    async def test_invalid_yaml_content(self, mock_objective_target):
        """Test handling of malformed YAML files."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        with patch("pathlib.Path.read_text") as mock_read, patch("yaml.safe_load") as mock_yaml:

            mock_read.return_value = "invalid yaml content"
            mock_yaml.side_effect = yaml.YAMLError("Invalid YAML")

            with pytest.raises(yaml.YAMLError):
                attack._load_prompt_from_yaml(yaml_filename="invalid.yaml")

    @pytest.mark.asyncio
    async def test_missing_memory_labels_attribute(self, mock_objective_target, mock_processing_model, sample_context):
        """Test handling when _memory_labels attribute is missing during KG extraction."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target, processing_model=mock_processing_model)

        # Ensure _memory_labels exists (it should be set by base class)
        if not hasattr(attack, "_memory_labels"):
            attack._memory_labels = {}

        with (
            patch.object(attack, "_load_prompt_from_yaml") as mock_load,
            patch.object(attack, "_format_few_shot_examples") as mock_format,
        ):

            mock_load.return_value = "KG prompt"
            mock_format.return_value = "examples"
            attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=None)

            with pytest.raises(RuntimeError, match="Failed to extract knowledge graph"):
                await attack._extract_knowledge_graph_async(context=sample_context)

    @pytest.mark.asyncio
    async def test_yaml_missing_value_key(self, mock_objective_target):
        """Test handling of YAML files missing the 'value' key."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        with patch("pathlib.Path.read_text") as mock_read, patch("yaml.safe_load") as mock_yaml:

            mock_read.return_value = "valid yaml"
            mock_yaml.return_value = {"other_key": "some value"}  # Missing 'value' key

            with pytest.raises(KeyError):
                attack._load_prompt_from_yaml(yaml_filename="no_value_key.yaml")

    @pytest.mark.asyncio
    async def test_prompt_normalizer_failure_during_attack(self, mock_objective_target, sample_context):
        """Test handling when prompt normalizer fails during attack execution."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        attack._prompt_normalizer.send_prompt_async = AsyncMock(
            side_effect=RuntimeError("Normalizer connection failed")
        )

        with patch.object(attack, "_prepare_examples_async") as mock_prepare:
            mock_prepare.return_value = "formatted examples"

            with pytest.raises(RuntimeError, match="Normalizer connection failed"):
                await attack._send_examples_to_target_async(formatted_examples="test examples", context=sample_context)

    @pytest.mark.asyncio
    async def test_scorer_exception_during_evaluation(self, mock_objective_target, sample_context, mock_response):
        """Test handling when scorer raises an exception."""
        mock_scorer = MagicMock()
        mock_scorer.scorer_type = "true_false"
        scoring_config = AttackScoringConfig(objective_scorer=mock_scorer)

        attack = AnecdoctorAttack(objective_target=mock_objective_target, attack_scoring_config=scoring_config)

        with patch(
            "pyrit.attacks.single_turn.anecdoctor.Scorer.score_response_with_objective_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Scorer evaluation failed"),
        ):
            with pytest.raises(RuntimeError, match="Scorer evaluation failed"):
                await attack._evaluate_response_async(response=mock_response, objective=sample_context.objective)

    @pytest.mark.asyncio
    async def test_knowledge_graph_extraction_with_empty_evaluation_data(
        self, mock_objective_target, mock_processing_model
    ):
        """Test KG extraction with empty evaluation data."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target, processing_model=mock_processing_model)

        context = AnecdoctorAttackContext(
            objective="test objective",
            memory_labels={"test": "label"},
            evaluation_data=[],  # Empty data
        )

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.return_value = "KG prompt"

            # Should still proceed but format empty examples
            with patch.object(attack, "_format_few_shot_examples") as mock_format:
                mock_format.return_value = "### examples\n"

                mock_kg_response = MagicMock()
                mock_kg_response.get_value.return_value = "Empty KG"
                attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=mock_kg_response)

                result = await attack._extract_knowledge_graph_async(context=context)
                assert result == "Empty KG"
                mock_format.assert_called_once_with(evaluation_data=[])


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_attack_with_very_long_evaluation_data(self, mock_objective_target, mock_response):
        """Test attack with extremely long evaluation data."""
        # Create very long evaluation data
        long_evaluation_data = [
            f"Claim {i}: This is a very long claim with lots of text. " * 100 + "Review: FALSE" for i in range(100)
        ]

        context = AnecdoctorAttackContext(
            objective="Generate content",
            memory_labels={"test": "edge_case"},
            evaluation_data=long_evaluation_data,
            language="english",
            content_type="article",
        )

        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=mock_response)

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.return_value = "System prompt"

            # Should handle large data without error
            result = await attack._perform_attack_async(context=context)
            assert result.outcome == AttackOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_attack_with_special_characters_in_data(self, mock_objective_target, mock_response):
        """Test attack with special characters in evaluation data."""
        special_data = [
            "Claim: Test with émojis 🚀💯 and unicode ñáéíóú. Review: FALSE",
            "Claim: Test with quotes \"double\" and 'single'. Review: TRUE",
            "Claim: Test with symbols @#$%^&*(){}[]|\\:;\"'<>?,./. Review: DISPUTED",
            "Claim: Test with newlines\nand\ttabs. Review: FALSE",
        ]

        context = AnecdoctorAttackContext(
            objective="Generate content",
            memory_labels={"test": "special_chars"},
            evaluation_data=special_data,
            language="english",
            content_type="tweet",
        )

        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=mock_response)

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.return_value = "System prompt"

            result = await attack._perform_attack_async(context=context)
            assert result.outcome == AttackOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_concurrent_attack_execution(self, mock_objective_target, sample_evaluation_data):
        """Test concurrent execution of multiple attacks."""
        import asyncio

        async def run_single_attack(attack_id: int):
            context = AnecdoctorAttackContext(
                objective=f"Objective {attack_id}",
                memory_labels={"attack_id": str(attack_id)},
                evaluation_data=sample_evaluation_data,
                language="english",
                content_type="tweet",
            )

            attack = AnecdoctorAttack(objective_target=mock_objective_target)

            mock_response = MagicMock()
            mock_response.get_piece.return_value = f"Response {attack_id}"

            with patch.object(attack._prompt_normalizer, "send_prompt_async", new_callable=AsyncMock) as mock_send:
                mock_send.return_value = mock_response

                with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
                    mock_load.return_value = f"System prompt {attack_id}"

                    await attack._setup_async(context=context)
                    return await attack._perform_attack_async(context=context)

        # Run multiple attacks concurrently
        results = await asyncio.gather(*[run_single_attack(i) for i in range(5)])

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.objective == f"Objective {i}"
            assert result.outcome == AttackOutcome.SUCCESS

    def test_context_memory_labels_persistence(self, sample_evaluation_data):
        """Test that memory labels are properly preserved throughout the attack."""
        original_labels = {"project": "test", "version": "1.0", "environment": "dev"}

        context = AnecdoctorAttackContext(
            objective="Test objective", memory_labels=original_labels.copy(), evaluation_data=sample_evaluation_data
        )

        # Verify labels are preserved
        assert context.memory_labels == original_labels

        # Modify labels to ensure they're not accidentally shared
        context.memory_labels["new_key"] = "new_value"
        assert "new_key" not in original_labels

    @pytest.mark.asyncio
    async def test_attack_identifier_uniqueness(self, mock_objective_target):
        """Test that each attack instance has a unique identifier."""
        attack1 = AnecdoctorAttack(objective_target=mock_objective_target)
        attack2 = AnecdoctorAttack(objective_target=mock_objective_target)

        id1 = attack1.get_identifier()
        id2 = attack2.get_identifier()

        # Verify identifier structure
        assert "__type__" in id1
        assert "__module__" in id1
        assert "id" in id1

        # Verify uniqueness
        assert id1["id"] != id2["id"]
        assert id1["__type__"] == id2["__type__"] == "AnecdoctorAttack"

    @pytest.mark.asyncio
    async def test_empty_string_evaluation_data_items(self, mock_objective_target, mock_response):
        """Test handling of empty strings in evaluation data."""
        evaluation_data_with_empty = [
            "Claim: Valid claim. Review: FALSE",
            "",  # Empty string
            "   ",  # Whitespace only
            "Claim: Another valid claim. Review: TRUE",
        ]

        context = AnecdoctorAttackContext(
            objective="Test objective",
            memory_labels={"test": "empty_strings"},
            evaluation_data=evaluation_data_with_empty,
        )

        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=mock_response)

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.return_value = "System prompt"

            # Should handle empty strings without error
            result = await attack._perform_attack_async(context=context)
            assert result.outcome == AttackOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_conversation_id_consistency(self, mock_objective_target, sample_context):
        """Test that conversation ID remains consistent throughout attack execution."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        original_id = sample_context.conversation_id

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.return_value = "System prompt"

            # Setup should generate new conversation ID
            await attack._setup_async(context=sample_context)
            setup_id = sample_context.conversation_id

            # ID should have changed during setup
            assert setup_id != original_id

            # Mock the rest of the attack flow
            mock_response = MagicMock()
            mock_response.get_piece.return_value = "Test response"

            with (
                patch.object(attack, "_prepare_examples_async") as mock_prepare,
                patch.object(attack, "_send_examples_to_target_async") as mock_send,
            ):

                mock_prepare.return_value = "examples"
                mock_send.return_value = mock_response

                result = await attack._perform_attack_async(context=sample_context)

                # Conversation ID should remain consistent
                assert result.conversation_id == setup_id
                assert sample_context.conversation_id == setup_id


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackScoringScenarios:
    """Tests for various scoring scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_scoring_with_no_objective_scores_returned(
        self, mock_objective_target, sample_context, mock_response
    ):
        """Test handling when scorer returns no objective scores."""
        mock_scorer = MagicMock()
        mock_scorer.scorer_type = "true_false"
        scoring_config = AttackScoringConfig(objective_scorer=mock_scorer)

        attack = AnecdoctorAttack(objective_target=mock_objective_target, attack_scoring_config=scoring_config)

        # Mock scorer to return empty objective_scores
        with patch(
            "pyrit.attacks.single_turn.anecdoctor.Scorer.score_response_with_objective_async",
            new_callable=AsyncMock,
            return_value={"objective_scores": []},  # Empty list
        ):
            result = await attack._evaluate_response_async(response=mock_response, objective=sample_context.objective)

            assert result is None

    @pytest.mark.asyncio
    async def test_scoring_with_auxiliary_scorers(self, mock_objective_target, sample_context, mock_response):
        """Test scoring with auxiliary scorers configured."""
        mock_objective_scorer = MagicMock()
        mock_objective_scorer.scorer_type = "true_false"
        mock_auxiliary_scorer = MagicMock()

        scoring_config = AttackScoringConfig(
            objective_scorer=mock_objective_scorer, auxiliary_scorers=[mock_auxiliary_scorer]
        )

        attack = AnecdoctorAttack(objective_target=mock_objective_target, attack_scoring_config=scoring_config)

        mock_score = MagicMock()
        mock_score.get_value.return_value = True

        with patch(
            "pyrit.attacks.single_turn.anecdoctor.Scorer.score_response_with_objective_async",
            new_callable=AsyncMock,
            return_value={"objective_scores": [mock_score]},
        ) as mock_scorer_call:

            result = await attack._evaluate_response_async(response=mock_response, objective=sample_context.objective)

            # Verify auxiliary scorers were passed to the scorer
            call_kwargs = mock_scorer_call.call_args.kwargs
            assert call_kwargs["auxiliary_scorers"] == [mock_auxiliary_scorer]
            assert result == mock_score

    def test_build_attack_result_with_all_fields(
        self, mock_objective_target, sample_context, mock_response, mock_score
    ):
        """Test building attack result with all possible fields populated."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        result = attack._build_attack_result(response=mock_response, score=mock_score, context=sample_context)

        # Verify all fields are properly set
        assert result.conversation_id == sample_context.conversation_id
        assert result.objective == sample_context.objective
        assert result.attack_identifier == attack.get_identifier()
        assert result.last_response == mock_response.get_piece()
        assert result.last_score == mock_score
        assert result.executed_turns == 1
        assert result.outcome in [AttackOutcome.SUCCESS, AttackOutcome.FAILURE]
        assert result.outcome_reason is not None

    def test_build_attack_result_with_none_response(self, mock_objective_target, sample_context):
        """Test building attack result when response is None."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        result = attack._build_attack_result(response=None, score=None, context=sample_context)

        assert result.last_response is None
        assert result.last_score is None
        assert result.outcome == AttackOutcome.FAILURE
        assert result.outcome_reason is not None
        assert "no response received" in result.outcome_reason


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackValidationEdgeCases:
    """Tests for validation edge cases and boundary conditions."""

    def test_validate_context_with_none_values(self, mock_objective_target):
        """Test validation with None values in context fields."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        # Create context with valid initialization, then modify field
        context = AnecdoctorAttackContext(
            objective="test objective",
            memory_labels={"test": "label"},
            evaluation_data=["test data"],
            language="english",
            content_type="tweet",
        )

        # Directly set the field to None to test validation
        context.language = ""  # Empty string instead of None to match validation logic

        with pytest.raises(ValueError, match="language must be provided"):
            attack._validate_context(context=context)

    def test_validate_context_with_whitespace_only_values(self, mock_objective_target):
        """Test validation with whitespace-only values."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        context = AnecdoctorAttackContext(
            objective="test objective",
            memory_labels={"test": "label"},
            evaluation_data=["test data"],
            language="   ",  # Whitespace only
            content_type="tweet",
        )

        # Current implementation only checks for empty string, not whitespace
        # This test documents current behavior
        attack._validate_context(context=context)  # Should pass

    def test_context_create_from_params_type_validation_edge_cases(self):
        """Test type validation edge cases in create_from_params."""

        # Test with None evaluation_data
        with pytest.raises(ValueError, match="evaluation_data must be a list"):
            AnecdoctorAttackContext.create_from_params(
                objective="test", prepended_conversation=[], memory_labels={}, evaluation_data=None
            )

        # Test with tuple instead of list (should fail)
        with pytest.raises(ValueError, match="evaluation_data must be a list"):
            AnecdoctorAttackContext.create_from_params(
                objective="test",
                prepended_conversation=[],
                memory_labels={},
                evaluation_data=("item1", "item2"),  # tuple instead of list
            )

    def test_context_create_from_params_with_extra_kwargs(self, sample_evaluation_data):
        """Test create_from_params ignores extra unknown kwargs."""
        context = AnecdoctorAttackContext.create_from_params(
            objective="test objective",
            prepended_conversation=[],
            memory_labels={"test": "label"},
            evaluation_data=sample_evaluation_data,
            unknown_param="should be ignored",
            another_unknown=123,
        )

        # Should create successfully, ignoring unknown parameters
        assert context.objective == "test objective"
        assert context.evaluation_data == sample_evaluation_data


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackLifecycle:
    """Tests for the complete attack lifecycle using execute_async."""

    @pytest.mark.asyncio
    async def test_execute_async_complete_flow_few_shot(
        self, mock_objective_target, sample_evaluation_data, mock_response
    ):
        """Test complete execute_async flow in few-shot mode."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)
        attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=mock_response)

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.return_value = "System prompt for {language} {type}"

            result = await attack.execute_async(
                objective="Generate misleading content",
                memory_labels={"test": "lifecycle"},
                evaluation_data=sample_evaluation_data,
                language="spanish",
                content_type="blog post",
            )

            # Verify successful execution
            assert result.outcome == AttackOutcome.SUCCESS
            assert result.objective == "Generate misleading content"
            assert result.executed_turns == 1
            assert result.last_response == mock_response.get_piece()

            # Verify system prompt was set with correct formatting
            mock_objective_target.set_system_prompt.assert_called_once()
            system_prompt_call = mock_objective_target.set_system_prompt.call_args
            assert "spanish" in system_prompt_call.kwargs["system_prompt"]
            assert "blog post" in system_prompt_call.kwargs["system_prompt"]

    @pytest.mark.asyncio
    async def test_execute_async_complete_flow_with_knowledge_graph(
        self, mock_objective_target, mock_processing_model, sample_evaluation_data, mock_response
    ):
        """Test complete execute_async flow with knowledge graph extraction."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target, processing_model=mock_processing_model)

        # Mock both KG extraction and final response
        mock_kg_response = MagicMock()
        mock_kg_response.get_value.return_value = "Extracted knowledge graph"

        def mock_send_async(*args, **kwargs):
            if kwargs["target"] == mock_processing_model:
                return mock_kg_response
            return mock_response

        attack._prompt_normalizer.send_prompt_async = AsyncMock(side_effect=mock_send_async)

        with patch.object(attack, "_load_prompt_from_yaml") as mock_load:
            mock_load.side_effect = lambda yaml_filename: f"Prompt from {yaml_filename}"

            result = await attack.execute_async(
                objective="Generate content using KG",
                memory_labels={"test": "kg_lifecycle"},
                evaluation_data=sample_evaluation_data,
                language="french",
                content_type="news article",
            )

            # Verify successful execution
            assert result.outcome == AttackOutcome.SUCCESS
            assert result.objective == "Generate content using KG"

            # Verify both models were configured
            mock_objective_target.set_system_prompt.assert_called_once()
            mock_processing_model.set_system_prompt.assert_called_once()

            # Verify prompt normalizer was called for both KG extraction and final generation
            assert attack._prompt_normalizer.send_prompt_async.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_async_validation_failure_prevents_execution(self, mock_objective_target):
        """Test that validation failure prevents attack execution."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        # Use empty evaluation_data which will fail validation in create_from_params
        with pytest.raises(ValueError, match="evaluation_data cannot be empty"):
            await attack.execute_async(
                objective="Test objective", evaluation_data=[]  # This will cause validation to fail
            )

    @pytest.mark.asyncio
    async def test_execute_async_setup_failure_triggers_teardown(self, mock_objective_target, sample_evaluation_data):
        """Test that setup failure still triggers teardown."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        with (
            patch.object(attack, "_validate_context") as mock_validate,
            patch.object(
                attack, "_setup_async", new_callable=AsyncMock, side_effect=RuntimeError("Setup failed")
            ) as mock_setup,
            patch.object(attack, "_perform_attack_async", new_callable=AsyncMock) as mock_perform,
            patch.object(attack, "_teardown_async", new_callable=AsyncMock) as mock_teardown,
        ):

            from pyrit.exceptions.exception_classes import AttackExecutionException

            with pytest.raises(AttackExecutionException, match="Unexpected error during attack execution"):
                await attack.execute_async(objective="Test objective", evaluation_data=sample_evaluation_data)

            # Verify setup was attempted and teardown was called
            mock_validate.assert_called_once()
            mock_setup.assert_called_once()
            mock_perform.assert_not_called()
            mock_teardown.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_async_perform_failure_triggers_teardown(self, mock_objective_target, sample_evaluation_data):
        """Test that perform failure still triggers teardown."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        with (
            patch.object(attack, "_validate_context") as mock_validate,
            patch.object(attack, "_setup_async", new_callable=AsyncMock) as mock_setup,
            patch.object(
                attack, "_perform_attack_async", new_callable=AsyncMock, side_effect=RuntimeError("Perform failed")
            ) as mock_perform,
            patch.object(attack, "_teardown_async", new_callable=AsyncMock) as mock_teardown,
        ):

            from pyrit.exceptions.exception_classes import AttackExecutionException

            with pytest.raises(AttackExecutionException, match="Unexpected error during attack execution"):
                await attack.execute_async(objective="Test objective", evaluation_data=sample_evaluation_data)

            # Verify all phases were attempted and teardown was called
            mock_validate.assert_called_once()
            mock_setup.assert_called_once()
            mock_perform.assert_called_once()
            mock_teardown.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_async_with_scoring_integration(
        self, mock_objective_target, sample_evaluation_data, mock_response, mock_score
    ):
        """Test execute_async with scoring integration."""
        mock_scorer = MagicMock()
        mock_scorer.scorer_type = "true_false"
        scoring_config = AttackScoringConfig(objective_scorer=mock_scorer)

        attack = AnecdoctorAttack(objective_target=mock_objective_target, attack_scoring_config=scoring_config)

        attack._prompt_normalizer.send_prompt_async = AsyncMock(return_value=mock_response)

        with (
            patch.object(attack, "_load_prompt_from_yaml") as mock_load,
            patch.object(
                attack, "_evaluate_response_async", new_callable=AsyncMock, return_value=mock_score
            ) as mock_eval,
        ):

            mock_load.return_value = "System prompt"

            result = await attack.execute_async(objective="Test with scoring", evaluation_data=sample_evaluation_data)

            # Verify scoring was integrated
            mock_eval.assert_called_once()
            assert result.last_score == mock_score

            # Verify outcome based on score
            if mock_score.get_value():
                assert result.outcome == AttackOutcome.SUCCESS
            else:
                assert result.outcome == AttackOutcome.FAILURE


@pytest.mark.usefixtures("patch_central_database")
class TestAnecdoctorAttackWarningMechanisms:
    """Tests for warning mechanisms and configuration validation."""

    def test_warn_if_set_with_refusal_scorer(self, mock_objective_target):
        """Test that setting unused refusal_scorer triggers a warning."""
        mock_refusal_scorer = MagicMock()
        scoring_config = AttackScoringConfig(refusal_scorer=mock_refusal_scorer)

        # Should initialize without error but may log warnings
        attack = AnecdoctorAttack(objective_target=mock_objective_target, attack_scoring_config=scoring_config)

        # Verify the attack was created successfully
        assert attack._objective_target == mock_objective_target
        # refusal_scorer should not be used in AnecdoctorAttack
        assert not hasattr(attack, "_refusal_scorer")

    def test_initialization_with_all_optional_configs(
        self, mock_objective_target, mock_processing_model, mock_prompt_normalizer
    ):
        """Test initialization with all optional configurations."""
        mock_request_converter = MagicMock()
        mock_response_converter = MagicMock()
        mock_auxiliary_scorer = MagicMock()
        mock_objective_scorer = MagicMock()
        mock_objective_scorer.scorer_type = "true_false"

        converter_config = AttackConverterConfig(
            request_converters=[mock_request_converter], response_converters=[mock_response_converter]
        )

        scoring_config = AttackScoringConfig(
            auxiliary_scorers=[mock_auxiliary_scorer], objective_scorer=mock_objective_scorer
        )

        attack = AnecdoctorAttack(
            objective_target=mock_objective_target,
            processing_model=mock_processing_model,
            attack_converter_config=converter_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Verify all configurations were properly set
        assert attack._objective_target == mock_objective_target
        assert attack._processing_model == mock_processing_model
        assert attack._prompt_normalizer == mock_prompt_normalizer
        assert attack._request_converters == [mock_request_converter]
        assert attack._response_converters == [mock_response_converter]
        assert attack._auxiliary_scorers == [mock_auxiliary_scorer]
        assert attack._objective_scorer == mock_objective_scorer

    def test_memory_labels_initialization_from_base_class(self, mock_objective_target):
        """Test that _memory_labels is properly initialized from base class."""
        attack = AnecdoctorAttack(objective_target=mock_objective_target)

        # The base class should initialize _memory_labels
        assert hasattr(attack, "_memory_labels")
        assert isinstance(attack._memory_labels, dict)
