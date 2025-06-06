# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.attacks.base.config import AttackConverterConfig, AttackScoringConfig
from pyrit.attacks.base.context import SingleTurnAttackContext
from pyrit.attacks.single_turn.prompt_injection import PromptInjectionAttack
from pyrit.exceptions.exception_classes import (
    AttackExecutionException,
    AttackValidationException,
)
from pyrit.models import (
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
    return SingleTurnAttackContext(
        objective="Test objective", conversation_id=str(uuid.uuid4()), max_attempts_on_failure=2
    )


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
class TestPromptInjectionAttackInitialization:
    """Tests for PromptInjectionAttack initialization and configuration"""

    def test_init_with_minimal_required_parameters(self, mock_target):
        attack = PromptInjectionAttack(objective_target=mock_target)

        assert attack._objective_target == mock_target
        assert attack._objective_scorer is None
        assert isinstance(attack._attack_converter_cfg, AttackConverterConfig)
        assert isinstance(attack._attack_scoring_cfg, AttackScoringConfig)
        assert isinstance(attack._prompt_normalizer, PromptNormalizer)

    def test_init_with_valid_true_false_scorer(self, mock_target, mock_true_false_scorer):
        attack = PromptInjectionAttack(objective_target=mock_target, objective_scorer=mock_true_false_scorer)

        assert attack._objective_scorer == mock_true_false_scorer

    def test_init_raises_error_for_non_true_false_scorer(self, mock_target, mock_non_true_false_scorer):
        with pytest.raises(ValueError, match="Objective scorer must be a true/false scorer"):
            PromptInjectionAttack(objective_target=mock_target, objective_scorer=mock_non_true_false_scorer)

    def test_init_with_all_custom_configurations(self, mock_target, mock_true_false_scorer, mock_prompt_normalizer):
        converter_cfg = AttackConverterConfig()
        scoring_cfg = AttackScoringConfig()

        attack = PromptInjectionAttack(
            objective_target=mock_target,
            objective_scorer=mock_true_false_scorer,
            attack_converter_cfg=converter_cfg,
            attack_scoring_cfg=scoring_cfg,
            prompt_normalizer=mock_prompt_normalizer,
        )

        assert attack._attack_converter_cfg == converter_cfg
        assert attack._attack_scoring_cfg == scoring_cfg
        assert attack._prompt_normalizer == mock_prompt_normalizer

    def test_conversation_manager_initialized_correctly(self, mock_target):
        attack = PromptInjectionAttack(objective_target=mock_target)

        assert attack._conversation_manager is not None
        assert hasattr(attack._conversation_manager, "update_conversation_state_async")


@pytest.mark.usefixtures("patch_central_database")
class TestContextValidation:
    """Tests for context validation logic"""

    @pytest.mark.parametrize(
        "objective,conversation_id,expected_error",
        [
            ("", str(uuid.uuid4()), "Attack objective must be provided"),
            ("Test objective", "", "Conversation ID must be provided"),
        ],
    )
    def test_validate_context_raises_errors(self, mock_target, objective, conversation_id, expected_error):
        attack = PromptInjectionAttack(objective_target=mock_target)
        context = SingleTurnAttackContext(objective=objective, conversation_id=conversation_id)

        with pytest.raises(ValueError, match=expected_error):
            attack._validate_context(context=context)

    def test_validate_context_with_complete_valid_context(self, mock_target, basic_context):
        attack = PromptInjectionAttack(objective_target=mock_target)
        attack._validate_context(context=basic_context)  # Should not raise

    def test_validate_context_with_additional_optional_fields(self, mock_target):
        attack = PromptInjectionAttack(objective_target=mock_target)
        context = SingleTurnAttackContext(
            objective="Test objective",
            conversation_id=str(uuid.uuid4()),
            max_attempts_on_failure=5,
            seed_prompt_group=SeedPromptGroup(prompts=[SeedPrompt(value="test", data_type="text")]),
            system_prompt="System prompt",
            metadata={"key": "value"},
        )

        attack._validate_context(context=context)  # Should not raise


@pytest.mark.usefixtures("patch_central_database")
class TestSetupPhase:
    """Tests for the setup phase of the attack"""

    @pytest.mark.asyncio
    async def test_setup_initializes_achieved_objective_to_false(self, mock_target, basic_context):
        attack = PromptInjectionAttack(objective_target=mock_target)
        attack._conversation_manager = MagicMock()
        attack._conversation_manager.update_conversation_state_async = AsyncMock()

        # Set to True to verify it gets reset
        basic_context.achieved_objective = True

        await attack._setup_async(context=basic_context)

        assert basic_context.achieved_objective is False

    @pytest.mark.asyncio
    async def test_setup_merges_memory_labels_correctly(self, mock_target, basic_context):
        attack = PromptInjectionAttack(objective_target=mock_target)

        # Add memory labels to both attack and context
        attack._memory_labels = {"strategy_label": "strategy_value", "common": "strategy"}
        basic_context.memory_labels = {"context_label": "context_value", "common": "context"}

        attack._conversation_manager = MagicMock()
        attack._conversation_manager.update_conversation_state_async = AsyncMock()

        await attack._setup_async(context=basic_context)

        # Context labels should override strategy labels for common keys
        assert basic_context.memory_labels == {
            "strategy_label": "strategy_value",
            "context_label": "context_value",
            "common": "context",
        }

    @pytest.mark.asyncio
    async def test_setup_updates_conversation_state_with_converters(self, mock_target, basic_context):
        from pyrit.prompt_normalizer.prompt_converter_configuration import (
            PromptConverterConfiguration,
        )

        converter_config = [PromptConverterConfiguration(converters=[])]
        attack = PromptInjectionAttack(
            objective_target=mock_target,
            attack_converter_cfg=AttackConverterConfig(request_converters=converter_config),
        )

        attack._conversation_manager = MagicMock()
        attack._conversation_manager.update_conversation_state_async = AsyncMock()

        await attack._setup_async(context=basic_context)

        attack._conversation_manager.update_conversation_state_async.assert_called_once_with(
            conversation_id=basic_context.conversation_id,
            prepended_conversation=basic_context.prepended_conversation,
            converter_configurations=converter_config,
        )


@pytest.mark.usefixtures("patch_central_database")
class TestPromptPreparation:
    """Tests for prompt preparation logic"""

    def test_get_prompt_group_uses_existing_seed_prompt_group(self, mock_target, basic_context):
        existing_group = SeedPromptGroup(prompts=[SeedPrompt(value="Existing prompt", data_type="text")])
        basic_context.seed_prompt_group = existing_group

        attack = PromptInjectionAttack(objective_target=mock_target)
        result = attack._get_prompt_group(basic_context)

        assert result == existing_group

    def test_get_prompt_group_creates_from_objective_when_no_seed_group(self, mock_target, basic_context):
        basic_context.seed_prompt_group = None
        basic_context.objective = "Custom objective text"

        attack = PromptInjectionAttack(objective_target=mock_target)
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

        attack = PromptInjectionAttack(
            objective_target=mock_target,
            prompt_normalizer=mock_prompt_normalizer,
            attack_converter_cfg=AttackConverterConfig(
                request_converters=request_converters, response_converters=response_converters
            ),
        )

        prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        basic_context.memory_labels = {"test": "label"}
        mock_response = MagicMock()
        mock_prompt_normalizer.send_prompt_async.return_value = mock_response

        result = await attack._send_prompt_to_target_async(prompt_group=prompt_group, context=basic_context)

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
        attack = PromptInjectionAttack(objective_target=mock_target, prompt_normalizer=mock_prompt_normalizer)

        prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        mock_prompt_normalizer.send_prompt_async.return_value = None

        result = await attack._send_prompt_to_target_async(prompt_group=prompt_group, context=basic_context)

        assert result is None


@pytest.mark.usefixtures("patch_central_database")
class TestResponseEvaluation:
    """Tests for response evaluation logic"""

    @pytest.mark.asyncio
    async def test_evaluate_response_with_objective_scorer_returns_score(
        self, mock_target, mock_true_false_scorer, sample_response, success_score
    ):
        attack = PromptInjectionAttack(objective_target=mock_target, objective_scorer=mock_true_false_scorer)

        with (
            patch("pyrit.score.scorer.Scorer.score_response_async", new=AsyncMock()) as mock_score_resp,
            patch(
                "pyrit.score.scorer.Scorer.score_response_until_success_async",
                new=AsyncMock(return_value=success_score),
            ) as mock_score_until,
        ):

            result = await attack._evaluate_response_async(response=sample_response, objective="Test objective")

            assert result == success_score

            # Verify auxiliary scorers were called
            mock_score_resp.assert_called_once_with(
                response=sample_response, scorers=attack._attack_scoring_cfg.auxiliary_scorers, role_filter="assistant"
            )

            # Verify objective scorer was called with task
            mock_score_until.assert_called_once_with(
                response=sample_response,
                scorers=[mock_true_false_scorer],
                role_filter="assistant",
                task="Test objective",
            )

    @pytest.mark.asyncio
    async def test_evaluate_response_without_objective_scorer_returns_none(self, mock_target, sample_response):
        attack = PromptInjectionAttack(objective_target=mock_target, objective_scorer=None)

        with patch("pyrit.score.scorer.Scorer.score_response_async", new=AsyncMock()) as mock_score_resp:
            result = await attack._evaluate_response_async(response=sample_response, objective="Test objective")

            assert result is None

            # Verify auxiliary scorers were still called
            mock_score_resp.assert_called_once_with(
                response=sample_response, scorers=attack._attack_scoring_cfg.auxiliary_scorers, role_filter="assistant"
            )

    @pytest.mark.asyncio
    async def test_evaluate_response_with_auxiliary_scorers(
        self, mock_target, mock_true_false_scorer, sample_response, success_score
    ):
        auxiliary_scorer = MagicMock(spec=Scorer)

        attack = PromptInjectionAttack(
            objective_target=mock_target,
            objective_scorer=mock_true_false_scorer,
            attack_scoring_cfg=AttackScoringConfig(auxiliary_scorers=[auxiliary_scorer]),
        )

        with (
            patch("pyrit.score.scorer.Scorer.score_response_async", new=AsyncMock()) as mock_score_resp,
            patch(
                "pyrit.score.scorer.Scorer.score_response_until_success_async",
                new=AsyncMock(return_value=success_score),
            ),
        ):

            await attack._evaluate_response_async(response=sample_response, objective="Test objective")

            # Verify auxiliary scorers were called with the configured scorer
            call_args = mock_score_resp.call_args
            assert call_args.kwargs["scorers"] == [auxiliary_scorer]


@pytest.mark.usefixtures("patch_central_database")
class TestAttackExecution:
    """Tests for the main attack execution logic"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "attempt_results,expected_attempts,expected_success",
        [
            # Success on first attempt
            ([("response", "success_score")], 1, True),
            # Success after 2 failures
            ([("response", "failure_score"), ("response", "failure_score"), ("response", "success_score")], 3, True),
            # All attempts fail
            ([("response", "failure_score"), ("response", "failure_score"), ("response", "failure_score")], 3, False),
            # All responses filtered
            ([(None, None), (None, None), (None, None)], 3, False),
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
        expected_success,
    ):
        attack = PromptInjectionAttack(objective_target=mock_target, objective_scorer=mock_true_false_scorer)

        # Create a copy of the context to avoid mutation between test runs
        test_context = SingleTurnAttackContext(
            objective=basic_context.objective,
            conversation_id=basic_context.conversation_id,
            max_attempts_on_failure=2,  # Allow 3 total attempts
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

        attack._send_prompt_to_target_async = AsyncMock(side_effect=responses)
        attack._evaluate_response_async = AsyncMock(side_effect=scores)
        attack._log_objective_status = MagicMock()

        # Execute the attack
        result = await attack._perform_attack_async(context=test_context)

        # Verify results
        assert attack._send_prompt_to_target_async.call_count == expected_attempts
        assert test_context.achieved_objective == expected_success
        assert result.achieved_objective == expected_success

        # Verify evaluation count (only called for non-None responses)
        expected_eval_count = sum(1 for resp, _ in attempt_results if resp == "response")
        assert attack._evaluate_response_async.call_count == expected_eval_count

    @pytest.mark.asyncio
    async def test_perform_attack_without_scorer_completes_after_first_response(
        self, mock_target, basic_context, sample_response
    ):
        attack = PromptInjectionAttack(objective_target=mock_target, objective_scorer=None)  # No scorer

        basic_context.max_attempts_on_failure = 5  # Many retries available

        # Mock the internal methods
        attack._get_prompt_group = MagicMock(
            return_value=SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        )
        attack._send_prompt_to_target_async = AsyncMock(return_value=sample_response)
        attack._evaluate_response_async = AsyncMock()
        attack._log_objective_status = MagicMock()

        # Execute the attack
        result = await attack._perform_attack_async(context=basic_context)

        # Verify completion without scoring
        assert basic_context.achieved_objective is False  # No scorer to determine success
        assert result.achieved_objective is False
        assert result.executed_turns == 1
        assert result.last_response == sample_response.get_piece()
        assert result.last_score is None

        # Verify only one attempt was made (no retries without scorer)
        attack._send_prompt_to_target_async.assert_called_once()
        attack._evaluate_response_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_perform_attack_without_scorer_retries_on_filtered_response(
        self, mock_target, basic_context, sample_response
    ):
        attack = PromptInjectionAttack(objective_target=mock_target, objective_scorer=None)  # No scorer

        basic_context.max_attempts_on_failure = 2

        # First attempt filtered, second succeeds
        attack._get_prompt_group = MagicMock(
            return_value=SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        )
        attack._send_prompt_to_target_async = AsyncMock(side_effect=[None, sample_response])
        attack._log_objective_status = MagicMock()

        # Execute the attack
        result = await attack._perform_attack_async(context=basic_context)

        # Verify completion after retry
        assert result.last_response == sample_response.get_piece()
        assert attack._send_prompt_to_target_async.call_count == 2


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

        attack = PromptInjectionAttack(
            objective_target=mock_target,
            attack_converter_cfg=AttackConverterConfig(request_converters=converter_config),
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.objective = input_text
        mock_prompt_normalizer.send_prompt_async.return_value = sample_response

        # Execute the attack
        await attack._perform_attack_async(context=basic_context)

        # Verify the converter configuration was passed
        call_args = mock_prompt_normalizer.send_prompt_async.call_args
        assert call_args.kwargs["request_converter_configurations"] == converter_config

    @pytest.mark.asyncio
    async def test_perform_attack_with_response_converters(
        self, mock_target, mock_prompt_normalizer, basic_context, sample_response
    ):
        response_converter = Base64Converter()
        converter_config = PromptConverterConfiguration.from_converters(converters=[response_converter])

        attack = PromptInjectionAttack(
            objective_target=mock_target,
            attack_converter_cfg=AttackConverterConfig(response_converters=converter_config),
            prompt_normalizer=mock_prompt_normalizer,
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_response

        # Execute the attack
        await attack._perform_attack_async(context=basic_context)

        # Verify the response converter configuration was passed
        call_args = mock_prompt_normalizer.send_prompt_async.call_args
        assert call_args.kwargs["response_converter_configurations"] == converter_config


@pytest.mark.usefixtures("patch_central_database")
class TestLogging:
    """Tests for logging functionality"""

    @pytest.mark.parametrize(
        "achieved_objective,expected_log",
        [
            (True, "achieved the objective"),
            (False, "has not achieved the objective"),
        ],
    )
    def test_log_objective_status(self, mock_target, basic_context, caplog, achieved_objective, expected_log):
        attack = PromptInjectionAttack(objective_target=mock_target)
        basic_context.achieved_objective = achieved_objective

        with caplog.at_level(logging.INFO):
            attack._log_objective_status(context=basic_context)

        assert expected_log in caplog.text.lower()


@pytest.mark.usefixtures("patch_central_database")
class TestAttackLifecycle:
    """Tests for the complete attack lifecycle (execute_async)"""

    @pytest.mark.asyncio
    async def test_execute_async_successful_lifecycle(self, mock_target, basic_context):
        attack = PromptInjectionAttack(objective_target=mock_target)

        # Mock all lifecycle methods
        attack._validate_context = MagicMock()
        attack._setup_async = AsyncMock()
        attack._perform_attack_async = AsyncMock(return_value="success_result")
        attack._teardown_async = AsyncMock()

        # Execute the complete lifecycle
        result = await attack.execute_async(context=basic_context)

        # Verify result and proper execution order
        assert result == "success_result"
        attack._validate_context.assert_called_once_with(context=basic_context)
        attack._setup_async.assert_called_once_with(context=basic_context)
        attack._perform_attack_async.assert_called_once_with(context=basic_context)
        attack._teardown_async.assert_called_once_with(context=basic_context)

    @pytest.mark.asyncio
    async def test_execute_async_validation_failure_prevents_execution(self, mock_target, basic_context):
        attack = PromptInjectionAttack(objective_target=mock_target)

        # Mock validation to fail
        attack._validate_context = MagicMock(side_effect=ValueError("Invalid context"))
        attack._setup_async = AsyncMock()
        attack._perform_attack_async = AsyncMock()
        attack._teardown_async = AsyncMock()

        # Should raise AttackValidationException
        with pytest.raises(AttackValidationException) as exc_info:
            await attack.execute_async(context=basic_context)

        # Verify error details
        assert "Context validation failed" in str(exc_info.value)

        # Verify only validation was attempted
        attack._validate_context.assert_called_once_with(context=basic_context)
        attack._setup_async.assert_not_called()
        attack._perform_attack_async.assert_not_called()
        attack._teardown_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_async_execution_error_still_calls_teardown(self, mock_target, basic_context):
        attack = PromptInjectionAttack(objective_target=mock_target)

        # Mock successful validation/setup but failed execution
        attack._validate_context = MagicMock()
        attack._setup_async = AsyncMock()
        attack._perform_attack_async = AsyncMock(side_effect=RuntimeError("Attack failed"))
        attack._teardown_async = AsyncMock()

        # Should raise AttackExecutionException
        with pytest.raises(AttackExecutionException) as exc_info:
            await attack.execute_async(context=basic_context)

        # Verify error details
        assert "Unexpected error during attack execution" in str(exc_info.value)

        # Verify all methods were called including teardown
        attack._validate_context.assert_called_once_with(context=basic_context)
        attack._setup_async.assert_called_once_with(context=basic_context)
        attack._perform_attack_async.assert_called_once_with(context=basic_context)
        attack._teardown_async.assert_called_once_with(context=basic_context)

    @pytest.mark.asyncio
    async def test_teardown_async_is_noop(self, mock_target, basic_context):
        attack = PromptInjectionAttack(objective_target=mock_target)

        # Should complete without error
        await attack._teardown_async(context=basic_context)
        # No assertions needed - we just want to ensure it runs without raising


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
        attack = PromptInjectionAttack(objective_target=mock_target, objective_scorer=mock_true_false_scorer)

        basic_context.max_attempts_on_failure = max_attempts

        # Mock successful response
        attack._get_prompt_group = MagicMock(
            return_value=SeedPromptGroup(prompts=[SeedPrompt(value="Test prompt", data_type="text")])
        )
        attack._send_prompt_to_target_async = AsyncMock(return_value=sample_response)
        attack._evaluate_response_async = AsyncMock(return_value=success_score)
        attack._log_objective_status = MagicMock()

        # Execute the attack
        result = await attack._perform_attack_async(context=basic_context)

        # Verify success with single attempt
        assert result.achieved_objective is True
        attack._send_prompt_to_target_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_attack_with_minimal_prompt_group(self, mock_target, basic_context, sample_response):
        attack = PromptInjectionAttack(objective_target=mock_target)

        # Set minimal prompt group with a single empty prompt
        minimal_group = SeedPromptGroup(prompts=[SeedPrompt(value="", data_type="text")])
        basic_context.seed_prompt_group = minimal_group

        attack._get_prompt_group = MagicMock(return_value=minimal_group)
        attack._send_prompt_to_target_async = AsyncMock(return_value=sample_response)
        attack._log_objective_status = MagicMock()

        # Execute the attack
        result = await attack._perform_attack_async(context=basic_context)

        # Verify it still executes
        assert result.executed_turns == 1
        attack._send_prompt_to_target_async.assert_called_with(prompt_group=minimal_group, context=basic_context)

    @pytest.mark.asyncio
    async def test_evaluate_response_handles_scorer_exception(
        self, mock_target, mock_true_false_scorer, sample_response
    ):
        attack = PromptInjectionAttack(objective_target=mock_target, objective_scorer=mock_true_false_scorer)

        with (
            patch("pyrit.score.scorer.Scorer.score_response_async", new=AsyncMock()),
            patch(
                "pyrit.score.scorer.Scorer.score_response_until_success_async",
                new=AsyncMock(side_effect=RuntimeError("Scorer error")),
            ),
        ):

            # Should propagate the exception
            with pytest.raises(RuntimeError, match="Scorer error"):
                await attack._evaluate_response_async(response=sample_response, objective="Test")

    def test_attack_has_unique_identifier(self, mock_target):
        attack1 = PromptInjectionAttack(objective_target=mock_target)
        attack2 = PromptInjectionAttack(objective_target=mock_target)

        id1 = attack1.get_identifier()
        id2 = attack2.get_identifier()

        # Verify identifier structure
        assert "__type__" in id1
        assert "__module__" in id1
        assert "id" in id1

        # Verify uniqueness
        assert id1["id"] != id2["id"]
        assert id1["__type__"] == id2["__type__"] == "PromptInjectionAttack"
