# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import (
    AttackConverterConfig,
    AttackScoringConfig,
    PromptSendingAttack,
    SingleTurnAttackContext,
)
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    ConversationType,
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.prompt_converter import Base64Converter, StringJoinConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer


@pytest.fixture
def mock_target():
    """Create a mock prompt target for testing"""
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = {"id": "mock_target_id"}
    return target


@pytest.fixture
def mock_true_false_scorer():
    """Create a mock true/false scorer for testing"""
    scorer = MagicMock(spec=Scorer)
    scorer.scorer_type = "true_false"
    scorer.score_text_async = AsyncMock()
    return scorer


@pytest.fixture
def mock_non_true_false_scorer():
    """Create a mock scorer that is not a true/false type"""
    scorer = MagicMock(spec=Scorer)
    scorer.scorer_type = "float_scale"
    return scorer


@pytest.fixture
def mock_prompt_normalizer():
    """Create a mock prompt normalizer for testing"""
    normalizer = MagicMock(spec=PromptNormalizer)
    normalizer.send_prompt_async = AsyncMock()
    return normalizer


@pytest.fixture
def basic_context():
    """Create a basic context for testing"""
    return SingleTurnAttackContext(objective="Test objective", conversation_id=str(uuid.uuid4()))


@pytest.fixture
def sample_response():
    """Create a sample response for testing"""
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="assistant", original_value="Test response", original_value_data_type="text")
        ]
    )


@pytest.fixture
def success_score():
    """Create a success score for testing"""
    return Score(
        score_type="true_false",
        score_value="true",
        score_category="test",
        score_value_description="Test success score",
        score_rationale="Test rationale for success",
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
    )


@pytest.fixture
def failure_score():
    """Create a failure score for testing"""
    return Score(
        score_type="true_false",
        score_value="false",
        score_category="test",
        score_value_description="Test failure score",
        score_rationale="Test rationale for failure",
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
    )


@pytest.mark.usefixtures("patch_central_database")
class TestPromptSendingAttackInitialization:
    """Tests for PromptSendingAttack initialization and configuration"""

    def test_init_with_minimal_required_parameters(self, mock_target):
        attack = PromptSendingAttack(objective_target=mock_target)

        assert attack._objective_target == mock_target
        assert attack._objective_scorer is None
        assert isinstance(attack._prompt_normalizer, PromptNormalizer)
        assert attack._max_attempts_on_failure == 0

    def test_init_with_valid_true_false_scorer(self, mock_target, mock_true_false_scorer):
        attack_scoring_config = AttackScoringConfig(objective_scorer=mock_true_false_scorer)
        attack = PromptSendingAttack(objective_target=mock_target, attack_scoring_config=attack_scoring_config)

        assert attack._objective_scorer == mock_true_false_scorer

    def test_init_raises_error_for_non_true_false_scorer(self, mock_target, mock_non_true_false_scorer):
        attack_scoring_config = AttackScoringConfig(objective_scorer=mock_non_true_false_scorer)
        with pytest.raises(ValueError, match="Objective scorer must be a true/false scorer"):
            PromptSendingAttack(objective_target=mock_target, attack_scoring_config=attack_scoring_config)

    def test_init_with_all_custom_configurations(self, mock_target, mock_true_false_scorer, mock_prompt_normalizer):
        converter_cfg = AttackConverterConfig()
        scoring_cfg = AttackScoringConfig(objective_scorer=mock_true_false_scorer)

        attack = PromptSendingAttack(
            objective_target=mock_target,
            attack_converter_config=converter_cfg,
            attack_scoring_config=scoring_cfg,
            prompt_normalizer=mock_prompt_normalizer,
            max_attempts_on_failure=5,
        )

        assert attack._request_converters == converter_cfg.request_converters
        assert attack._response_converters == converter_cfg.response_converters
        assert attack._objective_scorer == scoring_cfg.objective_scorer
        assert attack._auxiliary_scorers == scoring_cfg.auxiliary_scorers
        assert attack._prompt_normalizer == mock_prompt_normalizer
        assert attack._max_attempts_on_failure == 5

    def test_conversation_manager_initialized_correctly(self, mock_target):
        attack = PromptSendingAttack(objective_target=mock_target)

        assert attack._conversation_manager is not None
        assert hasattr(attack._conversation_manager, "update_conversation_state_async")

    def test_init_with_negative_max_attempts_raises_error(self, mock_target):
        with pytest.raises(ValueError, match="max_attempts_on_failure must be a non-negative integer"):
            PromptSendingAttack(objective_target=mock_target, max_attempts_on_failure=-1)


@pytest.mark.usefixtures("patch_central_database")
class TestContextValidation:
    """Tests for context validation logic"""

    @pytest.mark.parametrize(
        "objective,conversation_id,expected_error",
        [
            ("", str(uuid.uuid4()), "Attack objective must be provided and non-empty in the context"),
        ],
    )
    def test_validate_context_raises_errors(self, mock_target, objective, conversation_id, expected_error):
        attack = PromptSendingAttack(objective_target=mock_target)
        context = SingleTurnAttackContext(objective=objective, conversation_id=conversation_id)

        with pytest.raises(ValueError, match=expected_error):
            attack._validate_context(context=context)

    def test_validate_context_with_complete_valid_context(self, mock_target, basic_context):
        attack = PromptSendingAttack(objective_target=mock_target)
        attack._validate_context(context=basic_context)  # Should not raise

    def test_validate_context_with_additional_optional_fields(self, mock_target):
        attack = PromptSendingAttack(objective_target=mock_target)
        context = SingleTurnAttackContext(
            objective="Test objective",
            conversation_id=str(uuid.uuid4()),
            seed_prompt_group=SeedPromptGroup(prompts=[SeedPrompt(value="test", data_type="text")]),
            system_prompt="System prompt",
            metadata={"key": "value"},
        )

        attack._validate_context(context=context)  # Should not raise


@pytest.mark.usefixtures("patch_central_database")
class TestSetupPhase:
    """Tests for the setup phase of the attack"""

    @pytest.mark.asyncio
    async def test_setup_merges_memory_labels_correctly(self, mock_target, basic_context):
        attack = PromptSendingAttack(objective_target=mock_target)

        # Add memory labels to both attack and context
        attack._memory_labels = {"strategy_label": "strategy_value", "common": "strategy"}
        basic_context.memory_labels = {"context_label": "context_value", "common": "context"}

        attack._conversation_manager = MagicMock()
        attack._conversation_manager.update_conversation_state_async = AsyncMock()

        # Store original conversation_id
        original_conversation_id = basic_context.conversation_id

        await attack._setup_async(context=basic_context)

        # Context labels should override strategy labels for common keys
        assert basic_context.memory_labels == {
            "strategy_label": "strategy_value",
            "context_label": "context_value",
            "common": "context",
        }

        # Conversation ID should be replaced with a new UUID
        assert basic_context.conversation_id != original_conversation_id
        assert basic_context.conversation_id  # Should have a new conversation_id

    @pytest.mark.asyncio
    async def test_setup_updates_conversation_state_with_converters(self, mock_target, basic_context):
        from pyrit.prompt_normalizer.prompt_converter_configuration import (
            PromptConverterConfiguration,
        )

        converter_config = [PromptConverterConfiguration(converters=[])]
        attack = PromptSendingAttack(
            objective_target=mock_target,
            attack_converter_config=AttackConverterConfig(request_converters=converter_config),
        )

        attack._conversation_manager = MagicMock()
        attack._conversation_manager.update_conversation_state_async = AsyncMock()

        await attack._setup_async(context=basic_context)

        attack._conversation_manager.update_conversation_state_async.assert_called_once_with(
            target=mock_target,
            conversation_id=basic_context.conversation_id,
            prepended_conversation=basic_context.prepended_conversation,
            request_converters=converter_config,
            response_converters=[],
        )


@pytest.mark.usefixtures("patch_central_database")
class TestPromptPreparation:
    """Tests for prompt preparation logic"""

    def test_get_prompt_group_uses_existing_seed_prompt_group(self, mock_target, basic_context):
        existing_group = SeedPromptGroup(prompts=[SeedPrompt(value="Existing prompt", data_type="text")])
        basic_context.seed_prompt_group = existing_group

        attack = PromptSendingAttack(objective_target=mock_target)
        result = attack._get_prompt_group(basic_context)

        assert result == existing_group

    def test_get_prompt_group_creates_from_objective_when_no_seed_group(self, mock_target, basic_context):
        basic_context.seed_prompt_group = None
        basic_context.objective = "Custom objective text"

        attack = PromptSendingAttack(objective_target=mock_target)
        result = attack._get_prompt_group(basic_context)

        assert isinstance(result, SeedPromptGroup)
        assert len(result.prompts) == 1
        assert result.prompts[0].value == "Custom objective text"
        assert result.prompts[0].data_type == "text"


@pytest.mark.usefixtures("patch_central_database")
class TestPromptSending:
    """Tests for sending prompts to target"""

    @pytest.mark.asyncio
    async def test_send_prompt_to_target_with_all_configurations(
        self, mock_target, mock_prompt_normalizer, basic_context
    ):
        from pyrit.prompt_normalizer.prompt_converter_configuration import (
            PromptConverterConfiguration,
        )

        request_converters = [PromptConverterConfiguration(converters=[])]
        response_converters = [PromptConverterConfiguration(converters=[])]

        attack = PromptSendingAttack(
            objective_target=mock_target,
            prompt_normalizer=mock_prompt_normalizer,
            attack_converter_config=AttackConverterConfig(
                request_converters=request_converters, response_converters=response_converters
            ),
        )

        prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        basic_context.memory_labels = {"test": "label"}
        mock_response = MagicMock()
        mock_prompt_normalizer.send_prompt_async.return_value = mock_response

        result = await attack._send_prompt_to_objective_target_async(prompt_group=prompt_group, context=basic_context)

        assert result == mock_response

        # Verify all parameters were passed correctly
        call_args = mock_prompt_normalizer.send_prompt_async.call_args
        assert call_args.kwargs["seed_prompt_group"] == prompt_group
        assert call_args.kwargs["target"] == mock_target
        assert call_args.kwargs["conversation_id"] == basic_context.conversation_id
        assert call_args.kwargs["request_converter_configurations"] == request_converters
        assert call_args.kwargs["response_converter_configurations"] == response_converters
        assert call_args.kwargs["labels"] == {"test": "label"}
        assert "orchestrator_identifier" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_send_prompt_handles_none_response(self, mock_target, mock_prompt_normalizer, basic_context):
        attack = PromptSendingAttack(objective_target=mock_target, prompt_normalizer=mock_prompt_normalizer)

        prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        mock_prompt_normalizer.send_prompt_async.return_value = None

        result = await attack._send_prompt_to_objective_target_async(prompt_group=prompt_group, context=basic_context)

        assert result is None


@pytest.mark.usefixtures("patch_central_database")
class TestResponseEvaluation:
    """Tests for response evaluation logic"""

    @pytest.mark.asyncio
    async def test_evaluate_response_with_objective_scorer_returns_score(
        self, mock_target, mock_true_false_scorer, sample_response, success_score
    ):
        attack_scoring_config = AttackScoringConfig(objective_scorer=mock_true_false_scorer)
        attack = PromptSendingAttack(objective_target=mock_target, attack_scoring_config=attack_scoring_config)

        with patch(
            "pyrit.score.Scorer.score_response_with_objective_async",
            new_callable=AsyncMock,
            return_value={"auxiliary_scores": [], "objective_scores": [success_score]},
        ) as mock_score_method:

            result = await attack._evaluate_response_async(response=sample_response, objective="Test objective")

            assert result == success_score

            # Verify the scorer was called with correct parameters
            mock_score_method.assert_called_once_with(
                response=sample_response,
                auxiliary_scorers=attack._auxiliary_scorers,
                objective_scorers=[mock_true_false_scorer],
                role_filter="assistant",
                task="Test objective",
            )

    @pytest.mark.asyncio
    async def test_evaluate_response_without_objective_scorer_returns_none(self, mock_target, sample_response):
        attack = PromptSendingAttack(objective_target=mock_target, attack_scoring_config=None)

        with patch(
            "pyrit.score.Scorer.score_response_with_objective_async",
            new_callable=AsyncMock,
            return_value={"auxiliary_scores": [], "objective_scores": []},
        ) as mock_score_method:
            result = await attack._evaluate_response_async(response=sample_response, objective="Test objective")

            assert result is None

            # Verify the scorer was called with no objective scorers
            mock_score_method.assert_called_once_with(
                response=sample_response,
                auxiliary_scorers=attack._auxiliary_scorers,
                objective_scorers=None,
                role_filter="assistant",
                task="Test objective",
            )

    @pytest.mark.asyncio
    async def test_evaluate_response_with_auxiliary_scorers(
        self, mock_target, mock_true_false_scorer, sample_response, success_score
    ):
        auxiliary_scorer = MagicMock(spec=Scorer)
        auxiliary_score = Score(
            score_type="float_scale",
            score_value="0.8",
            score_category="test_auxiliary",
            score_value_description="Auxiliary score",
            score_rationale="Auxiliary rationale",
            score_metadata="{}",
            prompt_request_response_id=str(uuid.uuid4()),
        )

        attack = PromptSendingAttack(
            objective_target=mock_target,
            attack_scoring_config=AttackScoringConfig(
                objective_scorer=mock_true_false_scorer, auxiliary_scorers=[auxiliary_scorer]
            ),
        )

        with patch(
            "pyrit.score.Scorer.score_response_with_objective_async",
            new_callable=AsyncMock,
            return_value={"auxiliary_scores": [auxiliary_score], "objective_scores": [success_score]},
        ) as mock_score_method:

            result = await attack._evaluate_response_async(response=sample_response, objective="Test objective")

            # Only objective score is returned
            assert result == success_score

            # Verify auxiliary scorers were passed correctly
            mock_score_method.assert_called_once_with(
                response=sample_response,
                auxiliary_scorers=[auxiliary_scorer],
                objective_scorers=[mock_true_false_scorer],
                role_filter="assistant",
                task="Test objective",
            )


@pytest.mark.usefixtures("patch_central_database")
class TestAttackExecution:
    """Tests for the main attack execution logic"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "attempt_results,expected_attempts,expected_outcome,expected_outcome_reason,max_attempts",
        [
            # Success on first attempt
            (
                [("response", "success_score")],
                1,
                AttackOutcome.SUCCESS,
                "Objective achieved according to scorer",
                2,
            ),
            # Success after 2 failures
            (
                [("response", "failure_score"), ("response", "failure_score"), ("response", "success_score")],
                3,
                AttackOutcome.SUCCESS,
                "Objective achieved according to scorer",
                2,
            ),
            # All attempts fail
            (
                [("response", "failure_score"), ("response", "failure_score"), ("response", "failure_score")],
                3,
                AttackOutcome.FAILURE,
                "Failed to achieve objective after 3 attempts",
                2,
            ),
            # All responses filtered
            (
                [(None, None), (None, None), (None, None)],
                3,
                AttackOutcome.FAILURE,
                "All attempts were filtered or failed to get a response",
                2,
            ),
        ],
    )
    async def test_perform_attack_with_various_retry_scenarios(
        self,
        mock_target,
        mock_true_false_scorer,
        basic_context,
        sample_response,
        success_score,
        failure_score,
        attempt_results,
        expected_attempts,
        expected_outcome,
        expected_outcome_reason,
        max_attempts,
    ):
        attack_scoring_config = AttackScoringConfig(objective_scorer=mock_true_false_scorer)
        attack = PromptSendingAttack(
            objective_target=mock_target,
            attack_scoring_config=attack_scoring_config,
            max_attempts_on_failure=max_attempts,
        )

        # Mock the internal methods
        attack._get_prompt_group = MagicMock(
            return_value=SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        )

        # Setup side effects based on attempt_results
        responses = []
        scores = []
        for resp, score in attempt_results:
            responses.append(sample_response if resp == "response" else None)
            if score == "success_score":
                scores.append(success_score)
            elif score == "failure_score":
                scores.append(failure_score)
            else:
                scores.append(None)

        attack._send_prompt_to_objective_target_async = AsyncMock(side_effect=responses)
        attack._evaluate_response_async = AsyncMock(side_effect=scores)

        # Execute the attack
        result = await attack._perform_async(context=basic_context)

        # Verify results
        assert attack._send_prompt_to_objective_target_async.call_count == expected_attempts
        assert result.outcome == expected_outcome
        assert result.outcome_reason == expected_outcome_reason

        # Verify evaluation count (only called for non-None responses)
        expected_eval_count = sum(1 for resp, _ in attempt_results if resp == "response")
        assert attack._evaluate_response_async.call_count == expected_eval_count

    @pytest.mark.asyncio
    async def test_perform_attack_without_scorer_completes_after_first_response(
        self, mock_target, basic_context, sample_response
    ):
        attack = PromptSendingAttack(
            objective_target=mock_target,
            attack_scoring_config=None,
            max_attempts_on_failure=5,  # Many retries available
        )

        # Mock the internal methods
        attack._get_prompt_group = MagicMock(
            return_value=SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        )
        attack._send_prompt_to_objective_target_async = AsyncMock(return_value=sample_response)
        attack._evaluate_response_async = AsyncMock(return_value=None)

        # Execute the attack
        result = await attack._perform_async(context=basic_context)

        # Verify completion without scoring
        assert result.outcome == AttackOutcome.UNDETERMINED
        assert result.outcome_reason == "No objective scorer configured"
        assert result.executed_turns == 1
        assert result.last_response == sample_response.get_piece()
        assert result.last_score is None

        # Verify only one attempt was made (no retries without scorer)
        attack._send_prompt_to_objective_target_async.assert_called_once()

        # Verify that _evaluate_response_async was called even without objective scorer
        # This ensures auxiliary scores are still collected
        attack._evaluate_response_async.assert_called_once_with(
            response=sample_response, objective=basic_context.objective
        )

    @pytest.mark.asyncio
    async def test_perform_attack_without_scorer_retries_on_filtered_response(
        self, mock_target, basic_context, sample_response
    ):
        attack = PromptSendingAttack(
            objective_target=mock_target, attack_scoring_config=None, max_attempts_on_failure=2
        )

        # First attempt filtered, second succeeds
        attack._get_prompt_group = MagicMock(
            return_value=SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        )
        attack._send_prompt_to_objective_target_async = AsyncMock(side_effect=[None, sample_response])

        # Execute the attack
        result = await attack._perform_async(context=basic_context)

        # Verify completion after retry
        assert result.last_response == sample_response.get_piece()
        assert attack._send_prompt_to_objective_target_async.call_count == 2


@pytest.mark.usefixtures("patch_central_database")
class TestConverterIntegration:
    """Tests for converter integration"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "converters,input_text,expected_pattern",
        [
            # Single Base64 converter
            ([Base64Converter()], "Hello", "SGVsbG8="),
            # Single StringJoin converter
            ([StringJoinConverter(join_value="_")], "Hello", "H_e_l_l_o"),
            # Stacked converters: Base64 then StringJoin
            ([Base64Converter(), StringJoinConverter(join_value="_")], "Hello", "S_G_V_s_b_G_8_="),
            # Empty converter list
            ([], "Hello", "Hello"),
        ],
    )
    async def test_perform_attack_with_converters(
        self,
        mock_target,
        mock_prompt_normalizer,
        basic_context,
        sample_response,
        converters,
        input_text,
        expected_pattern,
    ):
        converter_config = PromptConverterConfiguration.from_converters(converters=converters)

        attack = PromptSendingAttack(
            objective_target=mock_target,
            attack_converter_config=AttackConverterConfig(request_converters=converter_config),
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.objective = input_text
        mock_prompt_normalizer.send_prompt_async.return_value = sample_response

        # Execute the attack
        await attack._perform_async(context=basic_context)

        # Verify the converter configuration was passed
        call_args = mock_prompt_normalizer.send_prompt_async.call_args
        assert call_args.kwargs["request_converter_configurations"] == converter_config

    @pytest.mark.asyncio
    async def test_perform_attack_with_response_converters(
        self, mock_target, mock_prompt_normalizer, basic_context, sample_response
    ):
        response_converter = Base64Converter()
        converter_config = PromptConverterConfiguration.from_converters(converters=[response_converter])

        attack = PromptSendingAttack(
            objective_target=mock_target,
            attack_converter_config=AttackConverterConfig(response_converters=converter_config),
            prompt_normalizer=mock_prompt_normalizer,
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_response

        # Execute the attack
        await attack._perform_async(context=basic_context)

        # Verify the response converter configuration was passed
        call_args = mock_prompt_normalizer.send_prompt_async.call_args
        assert call_args.kwargs["response_converter_configurations"] == converter_config


@pytest.mark.usefixtures("patch_central_database")
class TestDetermineAttackOutcome:
    """Tests for the _determine_attack_outcome method"""

    def test_determine_attack_outcome_success(self, mock_target, sample_response, success_score, basic_context):
        attack = PromptSendingAttack(objective_target=mock_target)
        attack._objective_scorer = MagicMock()

        outcome, reason = attack._determine_attack_outcome(
            response=sample_response, score=success_score, context=basic_context
        )

        assert outcome == AttackOutcome.SUCCESS
        assert reason == "Objective achieved according to scorer"

    def test_determine_attack_outcome_failure_with_response(
        self, mock_target, sample_response, failure_score, basic_context
    ):
        attack = PromptSendingAttack(objective_target=mock_target, max_attempts_on_failure=2)
        attack._objective_scorer = MagicMock()

        outcome, reason = attack._determine_attack_outcome(
            response=sample_response, score=failure_score, context=basic_context
        )

        assert outcome == AttackOutcome.FAILURE
        assert reason == "Failed to achieve objective after 3 attempts"

    def test_determine_attack_outcome_no_response(self, mock_target, basic_context):
        attack = PromptSendingAttack(objective_target=mock_target)
        attack._objective_scorer = MagicMock()

        outcome, reason = attack._determine_attack_outcome(response=None, score=None, context=basic_context)

        assert outcome == AttackOutcome.FAILURE
        assert reason == "All attempts were filtered or failed to get a response"

    def test_determine_attack_outcome_no_scorer(self, mock_target, sample_response, basic_context):
        attack = PromptSendingAttack(objective_target=mock_target)
        attack._objective_scorer = None

        outcome, reason = attack._determine_attack_outcome(response=sample_response, score=None, context=basic_context)

        assert outcome == AttackOutcome.UNDETERMINED
        assert reason == "No objective scorer configured"

    def test_determine_attack_outcome_no_scorer_no_response(self, mock_target, basic_context):
        """Test when no scorer is configured and no response is received"""
        attack = PromptSendingAttack(objective_target=mock_target)
        attack._objective_scorer = None

        outcome, reason = attack._determine_attack_outcome(response=None, score=None, context=basic_context)

        assert outcome == AttackOutcome.UNDETERMINED
        assert reason == "No objective scorer configured"

    def test_determine_attack_outcome_with_different_max_attempts(self, mock_target, sample_response, failure_score):
        """Test that the failure message correctly reflects the max_attempts_on_failure value"""

        for max_attempts in [0, 1, 5, 10]:
            attack = PromptSendingAttack(objective_target=mock_target, max_attempts_on_failure=max_attempts)
            attack._objective_scorer = MagicMock()

            context = SingleTurnAttackContext(objective="Test objective", conversation_id=str(uuid.uuid4()))

            outcome, reason = attack._determine_attack_outcome(
                response=sample_response, score=failure_score, context=context
            )

            assert outcome == AttackOutcome.FAILURE
            assert reason == f"Failed to achieve objective after {max_attempts + 1} attempts"

    def test_determine_attack_outcome_success_with_true_string_score(self, mock_target, sample_response, basic_context):
        """Test success detection with score value as string 'true'"""
        attack = PromptSendingAttack(objective_target=mock_target)
        attack._objective_scorer = MagicMock()

        true_score = Score(
            score_type="true_false",
            score_value="true",
            score_category="test",
            score_value_description="Success",
            score_rationale="Objective achieved",
            score_metadata="{}",
            prompt_request_response_id=str(uuid.uuid4()),
        )

        outcome, reason = attack._determine_attack_outcome(
            response=sample_response, score=true_score, context=basic_context
        )

        assert outcome == AttackOutcome.SUCCESS
        assert reason == "Objective achieved according to scorer"

    def test_determine_attack_outcome_failure_with_false_string_score(
        self, mock_target, sample_response, basic_context
    ):
        """Test failure detection with score value as string 'false'"""
        attack = PromptSendingAttack(objective_target=mock_target, max_attempts_on_failure=2)
        attack._objective_scorer = MagicMock()

        false_score = Score(
            score_type="true_false",
            score_value="false",
            score_category="test",
            score_value_description="Failure",
            score_rationale="Objective not achieved",
            score_metadata="{}",
            prompt_request_response_id=str(uuid.uuid4()),
        )

        outcome, reason = attack._determine_attack_outcome(
            response=sample_response, score=false_score, context=basic_context
        )

        assert outcome == AttackOutcome.FAILURE
        assert reason == "Failed to achieve objective after 3 attempts"

    def test_determine_attack_outcome_failure_with_uppercase_false_score(
        self, mock_target, sample_response, basic_context
    ):
        """Test failure detection with uppercase 'False' score value"""
        attack = PromptSendingAttack(objective_target=mock_target, max_attempts_on_failure=2)
        attack._objective_scorer = MagicMock()

        false_score = Score(
            score_type="true_false",
            score_value="False",
            score_category="test",
            score_value_description="Failure",
            score_rationale="Objective not achieved",
            score_metadata="{}",
            prompt_request_response_id=str(uuid.uuid4()),
        )

        outcome, reason = attack._determine_attack_outcome(
            response=sample_response, score=false_score, context=basic_context
        )

        assert outcome == AttackOutcome.FAILURE
        assert reason == "Failed to achieve objective after 3 attempts"

    def test_determine_attack_outcome_with_scorer_but_no_score(self, mock_target, sample_response, basic_context):
        """Test when scorer is configured but no score is returned (scorer failed)"""
        attack = PromptSendingAttack(objective_target=mock_target, max_attempts_on_failure=2)
        attack._objective_scorer = MagicMock()

        outcome, reason = attack._determine_attack_outcome(response=sample_response, score=None, context=basic_context)

        assert outcome == AttackOutcome.FAILURE
        assert reason == "Failed to achieve objective after 3 attempts"

    def test_determine_attack_outcome_edge_case_empty_response(self, mock_target, basic_context):
        """Test with an empty response object"""
        attack = PromptSendingAttack(objective_target=mock_target, max_attempts_on_failure=2)
        attack._objective_scorer = MagicMock()

        # Create an empty response
        empty_response = PromptRequestResponse(request_pieces=[])

        outcome, reason = attack._determine_attack_outcome(response=empty_response, score=None, context=basic_context)

        assert outcome == AttackOutcome.FAILURE
        assert reason == "Failed to achieve objective after 3 attempts"


@pytest.mark.usefixtures("patch_central_database")
class TestAttackLifecycle:
    """Tests for the complete attack lifecycle (execute_async)"""

    @pytest.mark.asyncio
    async def test_execute_async_successful_lifecycle(self, mock_target, basic_context):
        attack = PromptSendingAttack(objective_target=mock_target)

        # Mock all lifecycle methods
        attack._validate_context = MagicMock()
        attack._setup_async = AsyncMock()
        mock_result = AttackResult(
            conversation_id=basic_context.conversation_id,
            objective=basic_context.objective,
            attack_identifier=attack.get_identifier(),
            outcome=AttackOutcome.SUCCESS,
        )
        attack._perform_async = AsyncMock(return_value=mock_result)
        attack._teardown_async = AsyncMock()

        # Execute the complete lifecycle
        result = await attack.execute_with_context_async(context=basic_context)

        # Verify result and proper execution order
        assert result == mock_result
        attack._validate_context.assert_called_once_with(context=basic_context)
        attack._setup_async.assert_called_once_with(context=basic_context)
        attack._perform_async.assert_called_once_with(context=basic_context)
        attack._teardown_async.assert_called_once_with(context=basic_context)

    @pytest.mark.asyncio
    async def test_execute_async_validation_failure_prevents_execution(self, mock_target, basic_context):
        attack = PromptSendingAttack(objective_target=mock_target)

        # Mock validation to fail
        attack._validate_context = MagicMock(side_effect=ValueError("Invalid context"))
        attack._setup_async = AsyncMock()
        attack._perform_async = AsyncMock()
        attack._teardown_async = AsyncMock()

        # Should raise ValueError (validation error)
        with pytest.raises(ValueError) as exc_info:
            await attack.execute_with_context_async(context=basic_context)

        # Verify error details
        assert "Invalid context" in str(exc_info.value)

        # Verify only validation was attempted
        attack._validate_context.assert_called_once_with(context=basic_context)
        attack._setup_async.assert_not_called()
        attack._perform_async.assert_not_called()
        attack._teardown_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_async_execution_error_still_calls_teardown(self, mock_target, basic_context):
        attack = PromptSendingAttack(objective_target=mock_target)

        # Mock successful validation/setup but failed execution
        attack._validate_context = MagicMock()
        attack._setup_async = AsyncMock()
        attack._perform_async = AsyncMock(side_effect=RuntimeError("Attack failed"))
        attack._teardown_async = AsyncMock()

        # Should raise RuntimeError (execution error)
        with pytest.raises(RuntimeError) as exc_info:
            await attack.execute_with_context_async(context=basic_context)

        # Verify error details
        assert "Attack failed" in str(exc_info.value)

        # Verify all methods were called including teardown
        attack._validate_context.assert_called_once_with(context=basic_context)
        attack._setup_async.assert_called_once_with(context=basic_context)
        attack._perform_async.assert_called_once_with(context=basic_context)
        attack._teardown_async.assert_called_once_with(context=basic_context)

    @pytest.mark.asyncio
    async def test_teardown_async_is_noop(self, mock_target, basic_context):
        attack = PromptSendingAttack(objective_target=mock_target)

        # Should complete without error
        await attack._teardown_async(context=basic_context)
        # No assertions needed - we just want to ensure it runs without raising

    @pytest.mark.asyncio
    async def test_execute_async_with_parameters(self, mock_target, sample_response):
        """Test execute_async creates context using factory method and executes attack"""
        attack = PromptSendingAttack(objective_target=mock_target, max_attempts_on_failure=3)

        attack._validate_context = MagicMock()
        attack._setup_async = AsyncMock()
        mock_result = AttackResult(
            conversation_id="test-id",
            objective="Test objective",
            attack_identifier=attack.get_identifier(),
            outcome=AttackOutcome.SUCCESS,
            last_response=sample_response.get_piece(),
        )
        attack._perform_async = AsyncMock(return_value=mock_result)
        attack._teardown_async = AsyncMock()

        # Create test data
        seed_group = SeedPromptGroup(prompts=[SeedPrompt(value="test", data_type="text")])

        result = await attack.execute_async(
            objective="Test objective",
            prepended_conversation=[sample_response],
            memory_labels={"test": "label"},
            seed_prompt_group=seed_group,
            system_prompt="System prompt",
        )

        # Verify result
        assert result == mock_result

        # Verify context was created and passed properly
        call_args = attack._validate_context.call_args
        context = call_args.kwargs["context"]

        assert isinstance(context, SingleTurnAttackContext)
        assert context.objective == "Test objective"
        assert context.memory_labels == {"test": "label"}
        assert context.seed_prompt_group == seed_group
        assert context.system_prompt == "System prompt"

    @pytest.mark.asyncio
    async def test_execute_async_with_invalid_params_raises_error(self, mock_target):
        """Test execute_async raises error when invalid parameters are passed"""
        attack = PromptSendingAttack(objective_target=mock_target)

        # Test with invalid seed_prompt_group type
        with pytest.raises(TypeError, match="Parameter 'seed_prompt_group' must be of type SeedPromptGroup"):
            await attack.execute_async(
                objective="Test objective", seed_prompt_group="invalid_type"  # Should be SeedPromptGroup
            )

        # Test with invalid system_prompt type
        with pytest.raises(TypeError, match="Parameter 'system_prompt' must be of type str"):
            await attack.execute_async(objective="Test objective", system_prompt=123)  # Should be string


@pytest.mark.usefixtures("patch_central_database")
class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling scenarios"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_attempts", [0, 1, 5])
    async def test_perform_attack_with_various_max_attempts(
        self,
        mock_target,
        mock_true_false_scorer,
        basic_context,
        sample_response,
        success_score,
        max_attempts,
    ):
        attack_scoring_config = AttackScoringConfig(objective_scorer=mock_true_false_scorer)
        attack = PromptSendingAttack(
            objective_target=mock_target,
            attack_scoring_config=attack_scoring_config,
            max_attempts_on_failure=max_attempts,  # Set max_attempts here
        )

        # Mock successful response
        attack._get_prompt_group = MagicMock(
            return_value=SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        )
        attack._send_prompt_to_objective_target_async = AsyncMock(return_value=sample_response)
        attack._evaluate_response_async = AsyncMock(return_value=success_score)

        # Execute the attack
        result = await attack._perform_async(context=basic_context)

        # Verify success with single attempt
        assert result.outcome == AttackOutcome.SUCCESS
        attack._send_prompt_to_objective_target_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_attack_with_minimal_prompt_group(self, mock_target, basic_context, sample_response):
        attack = PromptSendingAttack(objective_target=mock_target)

        # Set minimal prompt group with a single empty prompt
        minimal_group = SeedPromptGroup(prompts=[SeedPrompt(value="", data_type="text")])
        basic_context.seed_prompt_group = minimal_group

        attack._get_prompt_group = MagicMock(return_value=minimal_group)
        attack._send_prompt_to_objective_target_async = AsyncMock(return_value=sample_response)

        # Execute the attack
        result = await attack._perform_async(context=basic_context)

        # Verify it still executes
        assert result.executed_turns == 1
        attack._send_prompt_to_objective_target_async.assert_called_with(
            prompt_group=minimal_group, context=basic_context
        )

    @pytest.mark.asyncio
    async def test_evaluate_response_handles_scorer_exception(
        self, mock_target, mock_true_false_scorer, sample_response
    ):
        attack_scoring_config = AttackScoringConfig(objective_scorer=mock_true_false_scorer)
        attack = PromptSendingAttack(objective_target=mock_target, attack_scoring_config=attack_scoring_config)

        with patch(
            "pyrit.score.Scorer.score_response_with_objective_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Scorer error"),
        ):

            # Should propagate the exception
            with pytest.raises(RuntimeError, match="Scorer error"):
                await attack._evaluate_response_async(response=sample_response, objective="Test")

    def test_attack_has_unique_identifier(self, mock_target):
        attack1 = PromptSendingAttack(objective_target=mock_target)
        attack2 = PromptSendingAttack(objective_target=mock_target)

        id1 = attack1.get_identifier()
        id2 = attack2.get_identifier()

        # Verify identifier structure
        assert "__type__" in id1
        assert "__module__" in id1
        assert "id" in id1

        # Verify uniqueness
        assert id1["id"] != id2["id"]
        assert id1["__type__"] == id2["__type__"] == "PromptSendingAttack"

    @pytest.mark.asyncio
    async def test_retry_stores_unsuccessful_conversation_and_updates_id(
        self, mock_target, mock_true_false_scorer, basic_context, sample_response, failure_score
    ):
        """Test that retries store unsuccessful conversations and update the conversation ID"""
        attack_scoring_config = AttackScoringConfig(objective_scorer=mock_true_false_scorer)
        attack = PromptSendingAttack(
            objective_target=mock_target,
            attack_scoring_config=attack_scoring_config,
            max_attempts_on_failure=2,  # Allow 2 retries
        )

        # Mock the internal methods
        attack._get_prompt_group = MagicMock(
            return_value=SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        )

        # Setup to return response on first two attempts but fail scoring, succeed on third
        responses = [sample_response, sample_response, sample_response]
        scores = [failure_score, failure_score, failure_score]  # All attempts fail

        attack._send_prompt_to_objective_target_async = AsyncMock(side_effect=responses)
        attack._evaluate_response_async = AsyncMock(side_effect=scores)

        # Store original conversation ID to verify it changes
        original_conversation_id = basic_context.conversation_id

        # Execute the attack
        result = await attack._perform_async(context=basic_context)

        # Verify that 3 attempts were made (initial + 2 retries)
        assert attack._send_prompt_to_objective_target_async.call_count == 3
        assert attack._evaluate_response_async.call_count == 3

        # Verify that related conversations were stored
        assert len(basic_context.related_conversations) == 2  # Two failed attempts stored

        # Verify the conversation references have correct type and different IDs
        related_conv_ids = {ref.conversation_id for ref in basic_context.related_conversations}
        assert len(related_conv_ids) == 2  # Two unique conversation IDs stored
        assert original_conversation_id in related_conv_ids  # Original ID should be in related conversations

        # Verify all stored conversations are marked as PRUNED
        for ref in basic_context.related_conversations:
            assert ref.conversation_type == ConversationType.PRUNED

        # Verify the final conversation ID is different from the original
        assert basic_context.conversation_id != original_conversation_id
        assert basic_context.conversation_id not in related_conv_ids

        # Verify attack still fails overall (since all attempts failed)
        assert result.outcome == AttackOutcome.FAILURE

        # Verify that the AttackResult includes the related conversations
        assert result.related_conversations == basic_context.related_conversations
        assert len(result.related_conversations) == 2
