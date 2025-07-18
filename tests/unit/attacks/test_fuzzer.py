# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional
from unittest.mock import AsyncMock, Mock, MagicMock, patch
import uuid

import pytest

from pyrit.attacks.fuzzer_attack import (
    FuzzerAttack,
    FuzzerAttackContext,
    FuzzerAttackResult,
    _PromptNode,
    _MCTSExplorer,
)
from pyrit.attacks.base.attack_config import AttackConverterConfig, AttackScoringConfig
from pyrit.exceptions import MissingPromptPlaceholderException
from pyrit.models import (
    AttackOutcome,
    PromptRequestResponse,
    PromptRequestPiece,
    Score,
    SeedPrompt,
)
from pyrit.prompt_converter import (
    FuzzerConverter,
    FuzzerExpandConverter,
    FuzzerShortenConverter,
    FuzzerRephraseConverter,
    FuzzerCrossOverConverter,
    ConverterResult,
)
from pyrit.exceptions import AttackValidationException
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.score import FloatScaleThresholdScorer, SelfAskScaleScorer


# Fixtures
@pytest.fixture
def mock_objective_target():
    """Mock prompt target for testing."""
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.send_prompt_batch_async = AsyncMock()
    return target


@pytest.fixture
def mock_scoring_target():
    """Mock chat target for scoring."""
    target = MagicMock(spec=PromptChatTarget)
    target.send_prompt_async = AsyncMock()
    return target


@pytest.fixture
def sample_prompts():
    """List of test prompts."""
    return ["Test prompt 1", "Test prompt 2", "Test prompt 3"]


@pytest.fixture
def sample_templates():
    """List of test templates with placeholder."""
    return [
        "Template 1: {{ prompt }}",
        "Template 2 with {{ prompt }} in middle",
        "{{ prompt }} at the beginning",
    ]


@pytest.fixture
def sample_converters(mock_scoring_target):
    """List of fuzzer converters."""
    return [
        FuzzerExpandConverter(converter_target=mock_scoring_target),
        FuzzerShortenConverter(converter_target=mock_scoring_target),
        FuzzerRephraseConverter(converter_target=mock_scoring_target),
    ]


@pytest.fixture
def basic_fuzzer_context():
    """Basic FuzzerAttackContext for testing."""
    return FuzzerAttackContext(
        objective="Test objective",
        memory_labels={"test": "label"},
    )


@pytest.fixture
def mock_objective_scorer():
    """Mock objective scorer."""
    scorer = MagicMock(spec=FloatScaleThresholdScorer)
    scorer.score_prompts_with_tasks_batch_async = AsyncMock()
    return scorer


@pytest.fixture
def mock_auxiliary_scorer():
    """Mock auxiliary scorer."""
    scorer = MagicMock()
    scorer.score_prompts_with_tasks_batch_async = AsyncMock()
    return scorer


@pytest.fixture
def attack_scoring_config(mock_objective_scorer, mock_auxiliary_scorer):
    """Attack scoring configuration."""
    return AttackScoringConfig(
        objective_scorer=mock_objective_scorer,
        auxiliary_scorers=[mock_auxiliary_scorer],
        successful_objective_threshold=0.8,
    )


@pytest.fixture
def attack_converter_config():
    """Attack converter configuration."""
    return AttackConverterConfig(
        request_converters=[],
        response_converters=[],
    )


@pytest.fixture
def mock_prompt_normalizer():
    """Mock prompt normalizer."""
    normalizer = MagicMock(spec=PromptNormalizer)
    normalizer.send_prompt_batch_to_target_async = AsyncMock()
    return normalizer


@pytest.fixture
def create_prompt_response():
    """Factory fixture to create PromptRequestResponse objects."""
    def _create(text: str = "Response text", conversation_id: Optional[str] = None):
        response = MagicMock(spec=PromptRequestResponse)
        piece = MagicMock(spec=PromptRequestPiece)
        piece.text = text
        piece.role = "assistant"
        piece.conversation_id = conversation_id or str(uuid.uuid4())
        response.request_pieces = [piece]
        return response
    return _create


@pytest.fixture
def create_score():
    """Factory fixture to create Score objects."""
    def _create(score_value: float = 0.5, score_type: str = "float_scale"):
        score = MagicMock(spec=Score)
        score.get_value.return_value = score_value
        score.score_type = score_type
        return score
    return _create


@pytest.fixture
def create_converter_result():
    """Factory fixture to create ConverterResult objects."""
    def _create(output_text: str, output_type: str = "text"):
        result = MagicMock(spec=ConverterResult)
        result.output_text = output_text
        result.output_type = output_type
        return result
    return _create


# Test Classes
@pytest.mark.usefixtures("patch_central_database")
class TestFuzzerAttackInitialization:
    """Tests the initialization and configuration of FuzzerAttack instances."""

    def test_init_with_required_parameters(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Verify all required parameters are properly stored and defaults are set."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Verify required parameters are stored
        assert fuzzer._objective_target == mock_objective_target
        assert fuzzer._prompts == sample_prompts
        assert fuzzer._prompt_templates == sample_templates
        assert fuzzer._template_converters == sample_converters
        assert fuzzer._objective_scorer == attack_scoring_config.objective_scorer

        # Verify default values
        assert fuzzer._batch_size == 10
        assert fuzzer._target_jailbreak_goal_count == 1
        assert fuzzer._max_query_limit == len(sample_templates) * len(sample_prompts) * 10
        assert isinstance(fuzzer._mcts_explorer, _MCTSExplorer)
        assert isinstance(fuzzer._prompt_normalizer, PromptNormalizer)

    def test_init_with_all_parameters(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        attack_converter_config,
        mock_prompt_normalizer,
    ):
        """Verify all parameters (required and optional) are properly initialized."""
        # Create fuzzer with all parameters
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_converter_config=attack_converter_config,
            attack_scoring_config=attack_scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
            frequency_weight=0.7,
            reward_penalty=0.2,
            minimum_reward=0.3,
            non_leaf_node_probability=0.15,
            batch_size=20,
            target_jailbreak_goal_count=5,
            max_query_limit=1000,
        )

        # Verify all parameters
        assert fuzzer._objective_target == mock_objective_target
        assert fuzzer._prompts == sample_prompts
        assert fuzzer._prompt_templates == sample_templates
        assert fuzzer._template_converters == sample_converters
        assert fuzzer._prompt_normalizer == mock_prompt_normalizer
        assert fuzzer._batch_size == 20
        assert fuzzer._target_jailbreak_goal_count == 5
        assert fuzzer._max_query_limit == 1000

        # Verify MCTS explorer is configured with custom values
        assert fuzzer._mcts_explorer.frequency_weight == 0.7
        assert fuzzer._mcts_explorer.reward_penalty == 0.2
        assert fuzzer._mcts_explorer.minimum_reward == 0.3
        assert fuzzer._mcts_explorer.non_leaf_node_probability == 0.15

    @pytest.mark.parametrize(
        "param_name,param_value,expected_error",
        [
            ("prompts", [], "The initial prompts cannot be empty"),
            ("prompt_templates", [], "The initial set of prompt templates cannot be empty"),
            ("template_converters", [], "Template converters cannot be empty"),
            ("batch_size", 0, "Batch size must be at least 1"),
            ("batch_size", -1, "Batch size must be at least 1"),
            ("max_query_limit", 2, "The query limit must be at least the number of prompts"),
        ],
    )
    def test_init_with_invalid_parameters(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        param_name,
        param_value,
        expected_error,
    ):
        """Test negative/zero values and empty lists raise appropriate errors."""
        # Prepare base parameters
        params = {
            "objective_target": mock_objective_target,
            "prompts": sample_prompts,
            "prompt_templates": sample_templates,
            "template_converters": sample_converters,
            "attack_scoring_config": attack_scoring_config,
        }

        # Override the parameter being tested
        params[param_name] = param_value

        # Verify error is raised
        with pytest.raises(ValueError, match=expected_error):
            FuzzerAttack(**params)

    def test_init_with_template_missing_placeholder(
        self,
        mock_objective_target,
        sample_prompts,
        sample_converters,
        attack_scoring_config,
    ):
        """Test templates without placeholder raise MissingPromptPlaceholderException."""
        invalid_templates = [
            "Template without placeholder",
            "Another template missing the marker",
        ]

        with pytest.raises(
            MissingPromptPlaceholderException,
            match="Template missing placeholder"
        ):
            FuzzerAttack(
                objective_target=mock_objective_target,
                prompts=sample_prompts,
                prompt_templates=invalid_templates,
                template_converters=sample_converters,
                attack_scoring_config=attack_scoring_config,
            )

    def test_init_with_no_objective_scorer(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
    ):
        """Test initialization fails when no objective scorer is provided."""
        # Create scoring config without objective scorer
        scoring_config = AttackScoringConfig(
            objective_scorer=None,
            auxiliary_scorers=[],
        )

        with pytest.raises(ValueError, match="Objective scorer must be provided"):
            FuzzerAttack(
                objective_target=mock_objective_target,
                prompts=sample_prompts,
                prompt_templates=sample_templates,
                template_converters=sample_converters,
                attack_scoring_config=scoring_config,
            )

    def test_init_with_custom_scoring_config(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_objective_scorer,
        mock_auxiliary_scorer,
    ):
        """Verify custom scoring configuration is properly set."""
        # Create custom scoring config
        scoring_config = AttackScoringConfig(
            objective_scorer=mock_objective_scorer,
            auxiliary_scorers=[mock_auxiliary_scorer],
            successful_objective_threshold=0.9,
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=scoring_config,
        )

        assert fuzzer._objective_scorer == mock_objective_scorer
        assert fuzzer._auxiliary_scorers == [mock_auxiliary_scorer]
        assert fuzzer._successful_objective_threshold == 0.9

    def test_init_with_custom_converter_config(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Verify custom converter configuration is properly set."""
        # Create mock converters
        request_converter = MagicMock()
        response_converter = MagicMock()

        converter_config = AttackConverterConfig(
            request_converters=[request_converter],
            response_converters=[response_converter],
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_converter_config=converter_config,
            attack_scoring_config=attack_scoring_config,
        )

        assert fuzzer._request_converters == [request_converter]
        assert fuzzer._response_converters == [response_converter]

    def test_default_query_limit_calculation(
        self,
        mock_objective_target,
        attack_scoring_config,
        sample_converters,
    ):
        """Test default query limit calculation (prompts * templates * 10)."""
        prompts = ["p1", "p2", "p3"]
        templates = ["t1: {{ prompt }}", "t2: {{ prompt }}"]

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=prompts,
            prompt_templates=templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        expected_limit = len(prompts) * len(templates) * 10
        assert fuzzer._max_query_limit == expected_limit

    @pytest.mark.parametrize(
        "frequency_weight,should_succeed",
        [
            (0.0, True),
            (0.5, True),
            (1.0, True),
            (-0.1, True), # No validation on frequency_weight
            (1.1, True), # No validation on frequency_weight
        ],
    )
    def test_frequency_weight_values(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        frequency_weight,
        should_succeed,
    ):
        """Test various frequency_weight values."""
        if should_succeed:
            fuzzer = FuzzerAttack(
                objective_target=mock_objective_target,
                prompts=sample_prompts,
                prompt_templates=sample_templates,
                template_converters=sample_converters,
                attack_scoring_config=attack_scoring_config,
                frequency_weight=frequency_weight,
            )
            assert fuzzer._mcts_explorer.frequency_weight == frequency_weight
        else:
            with pytest.raises(ValueError):
                FuzzerAttack(
                    objective_target=mock_objective_target,
                    prompts=sample_prompts,
                    prompt_templates=sample_templates,
                    template_converters=sample_converters,
                    attack_scoring_config=attack_scoring_config,
                    frequency_weight=frequency_weight,
                )


@pytest.mark.usefixtures("patch_central_database")
class TestFuzzerAttackFactoryMethods:
    """Tests the static factory methods for creating FuzzerAttack instances."""

    def test_with_default_scorer_creates_correct_scorer(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_scoring_target,
    ):
        """Verify SelfAskScaleScorer is created with correct parameters and FloatScaleThresholdScorer wrapper."""
        with patch("pyrit.attacks.fuzzer_attack.SelfAskScaleScorer") as mock_self_ask_scorer_class:
            with patch("pyrit.attacks.fuzzer_attack.FloatScaleThresholdScorer") as mock_threshold_scorer_class:
                # Setup mocks
                mock_self_ask_scorer = MagicMock()
                mock_threshold_scorer = MagicMock()
                mock_self_ask_scorer_class.return_value = mock_self_ask_scorer
                mock_threshold_scorer_class.return_value = mock_threshold_scorer

                # Create fuzzer with default scorer
                fuzzer = FuzzerAttack.with_default_scorer(
                    objective_target=mock_objective_target,
                    prompts=sample_prompts,
                    prompt_templates=sample_templates,
                    template_converters=sample_converters,
                    scoring_target=mock_scoring_target,
                )

                # Verify SelfAskScaleScorer was created with correct parameters
                # The actual enum values should be strings representing paths
                mock_self_ask_scorer_class.assert_called_once()
                call_args = mock_self_ask_scorer_class.call_args
                assert call_args.kwargs['chat_target'] == mock_scoring_target

                # Check that the enum paths are being accessed correctly
                assert 'scale_arguments_path' in call_args.kwargs
                assert 'system_prompt_path' in call_args.kwargs

                # Verify FloatScaleThresholdScorer was created with correct parameters
                mock_threshold_scorer_class.assert_called_once_with(
                    scorer=mock_self_ask_scorer,
                    threshold=0.8,
                )

                # Verify the fuzzer uses the threshold scorer
                assert fuzzer._objective_scorer == mock_threshold_scorer
                assert fuzzer._successful_objective_threshold == 0.8

    def test_with_default_scorer_parameters_passed_correctly(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_scoring_target,
        attack_converter_config,
        mock_prompt_normalizer,
    ):
        """Verify all parameters are passed through correctly to the FuzzerAttack constructor."""
        with patch("pyrit.attacks.fuzzer_attack.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer_attack.FloatScaleThresholdScorer"):
                # Create fuzzer with all optional parameters
                fuzzer = FuzzerAttack.with_default_scorer(
                    objective_target=mock_objective_target,
                    prompts=sample_prompts,
                    prompt_templates=sample_templates,
                    template_converters=sample_converters,
                    scoring_target=mock_scoring_target,
                    attack_converter_config=attack_converter_config,
                    prompt_normalizer=mock_prompt_normalizer,
                    frequency_weight=0.7,
                    reward_penalty=0.2,
                    minimum_reward=0.3,
                    non_leaf_node_probability=0.15,
                    batch_size=20,
                    target_jailbreak_goal_count=5,
                    max_query_limit=1000,
                )

                # Verify all parameters were set correctly
                assert fuzzer._objective_target == mock_objective_target
                assert fuzzer._prompts == sample_prompts
                assert fuzzer._prompt_templates == sample_templates
                assert fuzzer._template_converters == sample_converters
                assert fuzzer._prompt_normalizer == mock_prompt_normalizer
                assert fuzzer._request_converters == attack_converter_config.request_converters
                assert fuzzer._response_converters == attack_converter_config.response_converters
                assert fuzzer._batch_size == 20
                assert fuzzer._target_jailbreak_goal_count == 5
                assert fuzzer._max_query_limit == 1000

                # Verify MCTS parameters
                assert fuzzer._mcts_explorer.frequency_weight == 0.7
                assert fuzzer._mcts_explorer.reward_penalty == 0.2
                assert fuzzer._mcts_explorer.minimum_reward == 0.3
                assert fuzzer._mcts_explorer.non_leaf_node_probability == 0.15

    def test_with_default_scorer_calculates_query_limit(
        self,
        mock_objective_target,
        mock_scoring_target,
        sample_converters,
    ):
        """Test default query limit calculation (prompts * templates * 10) when not provided."""
        with patch("pyrit.attacks.fuzzer_attack.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer_attack.FloatScaleThresholdScorer"):
                # Test with different prompt and template counts
                test_cases = [
                    (["p1", "p2"], ["t1: {{ prompt }}", "t2: {{ prompt }}", "t3: {{ prompt }}"], 60),  # 2 * 3 * 10
                    (["p1", "p2", "p3"], ["t1: {{ prompt }}", "t2: {{ prompt }}"], 60),  # 3 * 2 * 10
                    (["p1"], ["t1: {{ prompt }}"], 10),  # 1 * 1 * 10
                ]

                for prompts, templates, expected_limit in test_cases:
                    fuzzer = FuzzerAttack.with_default_scorer(
                        objective_target=mock_objective_target,
                        prompts=prompts,
                        prompt_templates=templates,
                        template_converters=sample_converters,
                        scoring_target=mock_scoring_target,
                    )
                    assert fuzzer._max_query_limit == expected_limit

    def test_with_default_scorer_uses_provided_query_limit(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_scoring_target,
    ):
        """Test that provided query limit overrides default calculation."""
        with patch("pyrit.attacks.fuzzer_attack.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer_attack.FloatScaleThresholdScorer"):
                custom_limit = 500
                fuzzer = FuzzerAttack.with_default_scorer(
                    objective_target=mock_objective_target,
                    prompts=sample_prompts,
                    prompt_templates=sample_templates,
                    template_converters=sample_converters,
                    scoring_target=mock_scoring_target,
                    max_query_limit=custom_limit,
                )
                assert fuzzer._max_query_limit == custom_limit

    def test_with_default_scorer_invalid_parameters(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_scoring_target,
    ):
        """Test that invalid parameters still raise appropriate errors."""
        with patch("pyrit.attacks.fuzzer_attack.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer_attack.FloatScaleThresholdScorer"):
                # Test empty prompts
                with pytest.raises(ValueError, match="The initial prompts cannot be empty"):
                    FuzzerAttack.with_default_scorer(
                        objective_target=mock_objective_target,
                        prompts=[],
                        prompt_templates=sample_templates,
                        template_converters=sample_converters,
                        scoring_target=mock_scoring_target,
                    )

                # Test empty templates
                with pytest.raises(ValueError, match="The initial set of prompt templates cannot be empty"):
                    FuzzerAttack.with_default_scorer(
                        objective_target=mock_objective_target,
                        prompts=sample_prompts,
                        prompt_templates=[],
                        template_converters=sample_converters,
                        scoring_target=mock_scoring_target,
                    )

                # Test invalid batch size
                with pytest.raises(ValueError, match="Batch size must be at least 1"):
                    FuzzerAttack.with_default_scorer(
                        objective_target=mock_objective_target,
                        prompts=sample_prompts,
                        prompt_templates=sample_templates,
                        template_converters=sample_converters,
                        scoring_target=mock_scoring_target,
                        batch_size=0,
                    )

    def test_with_default_scorer_creates_prompt_normalizer(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_scoring_target,
    ):
        """Test that a default PromptNormalizer is created when not provided."""
        with patch("pyrit.attacks.fuzzer_attack.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer_attack.FloatScaleThresholdScorer"):
                fuzzer = FuzzerAttack.with_default_scorer(
                    objective_target=mock_objective_target,
                    prompts=sample_prompts,
                    prompt_templates=sample_templates,
                    template_converters=sample_converters,
                    scoring_target=mock_scoring_target,
                )
                assert isinstance(fuzzer._prompt_normalizer, PromptNormalizer)

    def test_with_default_scorer_uses_provided_normalizer(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_scoring_target,
        mock_prompt_normalizer,
    ):
        """Test that provided PromptNormalizer is used instead of creating a new one."""
        with patch("pyrit.attacks.fuzzer_attack.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer_attack.FloatScaleThresholdScorer"):
                fuzzer = FuzzerAttack.with_default_scorer(
                    objective_target=mock_objective_target,
                    prompts=sample_prompts,
                    prompt_templates=sample_templates,
                    template_converters=sample_converters,
                    scoring_target=mock_scoring_target,
                    prompt_normalizer=mock_prompt_normalizer,
                )
                assert fuzzer._prompt_normalizer == mock_prompt_normalizer


@pytest.mark.usefixtures("patch_central_database")
class TestFuzzerAttackValidation:
    """Tests context validation logic."""

    def test_validate_context_with_valid_context(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Should not raise any exceptions with valid context."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Should not raise any exception
        fuzzer._validate_context(context=basic_fuzzer_context)

    def test_validate_context_with_missing_objective(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Should raise ValueError with appropriate message when objective is missing."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Create context without objective
        context = FuzzerAttackContext(
            objective="",  # Empty objective
            memory_labels={"test": "label"},
        )

        with pytest.raises(ValueError, match="The attack objective must be set in the context"):
            fuzzer._validate_context(context=context)

    @pytest.mark.parametrize(
        "objective,should_pass",
        [
            ("Valid objective", True),
            ("A" * 1000, True), # Long objective
            ("123", True), # Numeric string
            ("", False), # Empty string
            ("   ", True), # Whitespace (passes validation but may not be meaningful)
            ("Special chars: !@#$%", True),
        ],
    )
    def test_validate_context_with_various_objectives(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        objective,
        should_pass,
    ):
        """Test validation with various objective values."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        context = FuzzerAttackContext(
            objective=objective,
            memory_labels={"test": "label"},
        )

        if should_pass:
            # Should not raise
            fuzzer._validate_context(context=context)
        else:
            with pytest.raises(ValueError, match="The attack objective must be set in the context"):
                fuzzer._validate_context(context=context)

    def test_validate_context_preserves_other_attributes(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Validation should not modify context attributes."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Create context with various attributes set
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={"test": "label", "another": "value"},
            total_target_query_count=10,
            total_jailbreak_count=2,
            jailbreak_conversation_ids=["id1", "id2"],
            executed_turns=5,
        )

        # Store original values
        original_objective = context.objective
        original_labels = context.memory_labels.copy()
        original_query_count = context.total_target_query_count
        original_jailbreak_count = context.total_jailbreak_count
        original_conversation_ids = context.jailbreak_conversation_ids.copy()
        original_turns = context.executed_turns

        fuzzer._validate_context(context=context)

        # Verify nothing changed
        assert context.objective == original_objective
        assert context.memory_labels == original_labels
        assert context.total_target_query_count == original_query_count
        assert context.total_jailbreak_count == original_jailbreak_count
        assert context.jailbreak_conversation_ids == original_conversation_ids
        assert context.executed_turns == original_turns

    def test_context_create_from_params(self):
        """Test the create_from_params factory method."""
        objective = "Test objective"
        prepended_conversation = []
        memory_labels = {"test": "label"}

        context = FuzzerAttackContext.create_from_params(
            objective=objective,
            prepended_conversation=prepended_conversation,
            memory_labels=memory_labels,
            # Extra kwargs should be ignored
            extra_param="should be ignored",
        )

        assert context.objective == objective
        assert context.memory_labels == memory_labels
        # Verify default values are set
        assert context.total_target_query_count == 0
        assert context.total_jailbreak_count == 0
        assert context.jailbreak_conversation_ids == []
        assert context.executed_turns == 0
        assert context.initial_prompt_nodes == []
        assert context.new_prompt_nodes == []
        assert context.mcts_selected_path == []
        assert context.last_choice_node is None

    def test_validate_context_called_during_attack_flow(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Verify that validate_context is called during the attack flow."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )
        
        with patch.object(fuzzer, '_validate_context', side_effect=ValueError("The attack objective must be set in the context.")) as mock_validate:
            # Create invalid context
            context = FuzzerAttackContext(
                objective="",  # Invalid
                memory_labels={},
            )

            # Try to execute attack - should call validate_context
            import asyncio
            with pytest.raises(AttackValidationException, match="Context validation failed: The attack objective must be set in the context"):
                asyncio.run(fuzzer.execute_with_context_async(context=context))

            # Verify validate was called
            mock_validate.assert_called_once()


@pytest.mark.usefixtures("patch_central_database")
class TestFuzzerAttackSetup:
    """Tests the setup phase of the attack."""

    @pytest.mark.asyncio
    async def test_setup_initializes_tracking_state(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Verify all counters are initialized to 0 and empty lists are initialized."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Run setup
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Verify all counters initialized to 0
        assert basic_fuzzer_context.total_target_query_count == 0
        assert basic_fuzzer_context.total_jailbreak_count == 0
        assert basic_fuzzer_context.executed_turns == 0

        # Verify empty lists initialized
        assert basic_fuzzer_context.jailbreak_conversation_ids == []
        assert basic_fuzzer_context.new_prompt_nodes == []
        assert basic_fuzzer_context.mcts_selected_path == []
        assert basic_fuzzer_context.last_choice_node is None

    @pytest.mark.asyncio
    async def test_setup_creates_initial_prompt_nodes(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Verify nodes are created for each template with correct properties."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Verify nodes created for each template
        assert len(basic_fuzzer_context.initial_prompt_nodes) == len(sample_templates)

        # Verify node properties
        for i, node in enumerate(basic_fuzzer_context.initial_prompt_nodes):
            assert isinstance(node, _PromptNode)
            assert node.template == sample_templates[i]
            assert node.level == 0 # Root nodes
            assert node.parent is None
            assert node.visited_num == 0
            assert node.rewards == 0
            assert node.children == []
            assert node.id is not None # UUID should be assigned

    @pytest.mark.asyncio
    async def test_setup_combines_memory_labels(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test memory label merging from context and attack instance."""
        # Create fuzzer with instance memory labels
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )
        
        # Set instance memory labels
        fuzzer._memory_labels = {"instance_key": "instance_value", "shared_key": "instance_value"}

        # Create context with different memory labels
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={"context_key": "context_value", "shared_key": "context_value"},
        )

        await fuzzer._setup_async(context=context)

        # Verify labels are combined with context taking precedence
        assert context.memory_labels == {
            "instance_key": "instance_value",
            "context_key": "context_value",
            "shared_key": "context_value", # Context value takes precedence
        }

    @pytest.mark.asyncio
    async def test_setup_initializes_mcts_structures(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Verify tree structures are properly initialized."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Run setup
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Verify MCTS structures
        assert isinstance(basic_fuzzer_context.initial_prompt_nodes, list)
        assert isinstance(basic_fuzzer_context.new_prompt_nodes, list)
        assert isinstance(basic_fuzzer_context.mcts_selected_path, list)
        assert len(basic_fuzzer_context.initial_prompt_nodes) > 0
        assert len(basic_fuzzer_context.new_prompt_nodes) == 0
        assert len(basic_fuzzer_context.mcts_selected_path) == 0

    @pytest.mark.asyncio
    async def test_setup_preserves_existing_context_objective(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Verify that setup preserves the objective from context."""
        original_objective = "Original test objective"
        context = FuzzerAttackContext(
            objective=original_objective,
            memory_labels={"test": "label"},
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Run setup
        await fuzzer._setup_async(context=context)

        # Verify objective is preserved
        assert context.objective == original_objective

    @pytest.mark.asyncio
    async def test_setup_resets_state_on_multiple_calls(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Verify that calling setup multiple times properly resets state."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # First setup
        await fuzzer._setup_async(context=basic_fuzzer_context)
        
        # Modify state
        basic_fuzzer_context.total_target_query_count = 10
        basic_fuzzer_context.total_jailbreak_count = 5
        basic_fuzzer_context.jailbreak_conversation_ids = ["id1", "id2"]
        basic_fuzzer_context.executed_turns = 3
        
        # Get initial nodes references
        first_initial_nodes = basic_fuzzer_context.initial_prompt_nodes

        # Second setup
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Verify state is reset
        assert basic_fuzzer_context.total_target_query_count == 0
        assert basic_fuzzer_context.total_jailbreak_count == 0
        assert basic_fuzzer_context.jailbreak_conversation_ids == []
        assert basic_fuzzer_context.executed_turns == 0

        # Verify new nodes are created (different instances)
        assert len(basic_fuzzer_context.initial_prompt_nodes) == len(first_initial_nodes)
        for i in range(len(first_initial_nodes)):
            assert basic_fuzzer_context.initial_prompt_nodes[i] is not first_initial_nodes[i]
            assert basic_fuzzer_context.initial_prompt_nodes[i].id != first_initial_nodes[i].id

    @pytest.mark.asyncio
    async def test_setup_with_empty_memory_labels(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test setup with no memory labels in context."""
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={},  # Empty labels
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Run setup
        await fuzzer._setup_async(context=context)

        # Should not raise and memory_labels should remain empty or contain instance labels
        assert isinstance(context.memory_labels, dict)

    @pytest.mark.parametrize(
        "num_templates",
        [1, 5, 10, 50],
    )
    @pytest.mark.asyncio
    async def test_setup_scales_with_template_count(
        self,
        mock_objective_target,
        sample_prompts,
        sample_converters,
        attack_scoring_config,
        num_templates,
    ):
        """Test setup correctly handles different numbers of templates."""
        # Generate templates
        templates = [f"Template {i}: {{{{ prompt }}}}" for i in range(num_templates)]

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={},
        )

        # Run setup
        await fuzzer._setup_async(context=context)

        # Verify correct number of nodes created
        assert len(context.initial_prompt_nodes) == num_templates
        
        # Verify each node has unique ID
        node_ids = {node.id for node in context.initial_prompt_nodes}
        assert len(node_ids) == num_templates  # All IDs should be unique

    @pytest.mark.asyncio
    async def test_setup_node_independence(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Verify that initial nodes are independent (no shared state)."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Run setup
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Modify one node
        basic_fuzzer_context.initial_prompt_nodes[0].visited_num = 10
        basic_fuzzer_context.initial_prompt_nodes[0].rewards = 5.0

        # Verify other nodes are not affected
        for i in range(1, len(basic_fuzzer_context.initial_prompt_nodes)):
            assert basic_fuzzer_context.initial_prompt_nodes[i].visited_num == 0
            assert basic_fuzzer_context.initial_prompt_nodes[i].rewards == 0


@pytest.mark.usefixtures("patch_central_database")
class TestFuzzerAttackExecution:
    """Tests the main attack execution flow."""

    @pytest.mark.asyncio
    async def test_perform_attack_successful_jailbreak(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_prompt_response,
        create_score,
        create_converter_result,  
    ):
        """Test successful attack execution that finds a jailbreak."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            batch_size=10,
            target_jailbreak_goal_count=1,
            max_query_limit=100,
        )

        # Mock converter to return a valid template - need to mock update method
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(
                return_value=create_converter_result("Converted: {{ prompt }}")
            )

        # Mock normalizer to return responses
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
            return_value=responses
        )

        # Mock scorer to return high scores (jailbreak)
        high_scores = [create_score(0.9) for _ in range(len(sample_prompts))]
        attack_scoring_config.objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(
            return_value=high_scores
        )

        # Run setup first
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Execute attack
        result = await fuzzer._perform_attack_async(context=basic_fuzzer_context)

        # Verify result
        assert isinstance(result, FuzzerAttackResult)
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.outcome_reason is not None
        assert "Found 3 jailbreaks" in result.outcome_reason  # 3 prompts scored high
        assert result.total_queries == len(sample_prompts)
        assert result.templates_explored >= 1
        assert len(result.jailbreak_conversation_ids) == len(sample_prompts)

    @pytest.mark.asyncio
    async def test_perform_attack_query_limit_reached(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_prompt_response,
        create_score,
        create_converter_result,  
    ):
        """Test attack stops when query limit is reached."""
        # Set low query limit
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            target_jailbreak_goal_count=10,  # High target
            max_query_limit=len(sample_prompts),  # Only allow one iteration
        )

        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(
                return_value=create_converter_result("Converted: {{ prompt }}")
            )

        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
            return_value=responses
        )

        # Mock scorer to return low scores (no jailbreak)
        low_scores = [create_score(0.3) for _ in range(len(sample_prompts))]
        attack_scoring_config.objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(
            return_value=low_scores
        )

        # Run setup first
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Execute attack
        result = await fuzzer._perform_attack_async(context=basic_fuzzer_context)

        # Verify result
        assert result.outcome == AttackOutcome.FAILURE
        assert result.outcome_reason is not None
        assert "Query limit" in result.outcome_reason
        assert result.total_queries == len(sample_prompts)
        assert basic_fuzzer_context.total_jailbreak_count == 0

    @pytest.mark.asyncio
    async def test_perform_attack_multiple_iterations(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_prompt_response,
        create_score,
        create_converter_result,  
    ):
        """Test attack runs multiple iterations before finding jailbreak."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            target_jailbreak_goal_count=1,
            max_query_limit=100,
        )

        # Mock converter - need to mock update method
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(
                return_value=create_converter_result("Converted: {{ prompt }}")
            )

        # Mock normalizer
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
            return_value=responses
        )

        # Mock scorer to return low scores first, then high
        call_count = 0
        def score_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # First two iterations fail
                return [create_score(0.3) for _ in range(len(sample_prompts))]
            else:  # Third iteration succeeds
                return [create_score(0.9) for _ in range(len(sample_prompts))]

        attack_scoring_config.objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(
            side_effect=score_side_effect
        )

        # Run setup first
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Execute attack
        result = await fuzzer._perform_attack_async(context=basic_fuzzer_context)

        # Verify result
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.executed_turns == 3
        assert result.total_queries == len(sample_prompts) * 3

    @pytest.mark.asyncio
    async def test_execute_attack_iteration_converter_error(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Test handling of converter errors during iteration."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(
                side_effect=MissingPromptPlaceholderException()
            )

        # Run setup first
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Execute one iteration should raise
        with pytest.raises(MissingPromptPlaceholderException):
            await fuzzer._execute_attack_iteration_async(basic_fuzzer_context)

    @pytest.mark.asyncio
    async def test_mcts_node_selection_and_update(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_prompt_response,
        create_score,
        create_converter_result,  
    ):
        """Test MCTS node selection and reward update logic."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            frequency_weight=0.5,
            reward_penalty=0.1,
        )

        # Mock converter - need to mock update method
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(
                return_value=create_converter_result("Converted: {{ prompt }}")
            )

        # Mock normalizer
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
            return_value=responses
        )

        # Mock scorer
        high_scores = [create_score(0.9) for _ in range(len(sample_prompts))]
        attack_scoring_config.objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(
            return_value=high_scores
        )

        # Run setup
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Execute one iteration
        await fuzzer._execute_attack_iteration_async(basic_fuzzer_context)

        # Verify MCTS state
        assert basic_fuzzer_context.executed_turns == 1
        assert len(basic_fuzzer_context.mcts_selected_path) > 0
        
        # Check that selected node has updated visit count and rewards
        selected_node = basic_fuzzer_context.last_choice_node
        assert selected_node.visited_num == 1
        assert selected_node.rewards > 0

        # Verify successful template was added
        assert len(basic_fuzzer_context.new_prompt_nodes) == 1

    @pytest.mark.asyncio
    async def test_auxiliary_scorers_called(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_objective_scorer,
        mock_auxiliary_scorer,
        basic_fuzzer_context,
        create_prompt_response,
        create_score,
    ):
        """Test that auxiliary scorers are called during scoring."""
        # Create scoring config with auxiliary scorer
        scoring_config = AttackScoringConfig(
            objective_scorer=mock_objective_scorer,
            auxiliary_scorers=[mock_auxiliary_scorer],
            successful_objective_threshold=0.8,
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=scoring_config,
        )

        # Mock responses
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]

        # Mock objective scorer
        mock_objective_scorer.score_prompts_with_tasks_batch_async.return_value = [
            create_score(0.5) for _ in range(len(sample_prompts))
        ]

        # Score responses
        await fuzzer._score_responses_async(responses=responses)

        # Verify both scorers were called
        mock_objective_scorer.score_prompts_with_tasks_batch_async.assert_called_once()
        mock_auxiliary_scorer.score_prompts_with_tasks_batch_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_processing(
        self,
        mock_objective_target,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_prompt_response,
    ):
        """Test that prompts are processed in batches according to batch_size."""
        # Use many prompts to test batching
        many_prompts = [f"Prompt {i}" for i in range(25)]
        batch_size = 10

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=many_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            batch_size=batch_size,
        )

        # Mock normalizer
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
            return_value=[create_prompt_response(f"Response {i}") for i in range(len(many_prompts))]
        )

        # Send prompts
        await fuzzer._send_prompts_to_target_async(
            context=basic_fuzzer_context,
            prompts=many_prompts
        )

        # Verify batch_size was passed to normalizer
        call_args = fuzzer._prompt_normalizer.send_prompt_batch_to_target_async.call_args
        assert call_args.kwargs['batch_size'] == batch_size

    @pytest.mark.asyncio
    async def test_score_normalization(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_prompt_response,
    ):
        """Test score normalization for different score types."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Test different score value types
        test_cases = [
            (True, 1.0),
            (False, 0.0),
            (1, 1.0),
            (0, 0.0),
            (0.5, 0.5),
            (1.5, 1.0),  # Clamped to 1.0
            (-0.5, 0.0),  # Clamped to 0.0
            ("invalid", 0.0),  # Invalid type defaults to 0.0
        ]

        for score_value, expected_normalized in test_cases:
            normalized = fuzzer._normalize_score_to_float(score_value)
            assert normalized == expected_normalized

    @pytest.mark.asyncio
    async def test_empty_response_handling(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Test handling of empty responses from target."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Test with empty responses
        scores = await fuzzer._score_responses_async(responses=[])
        assert scores == []

    @pytest.mark.asyncio
    async def test_converter_placeholder_validation(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_converter_result,  
    ):
        """Test that converter output is validated for placeholder presence."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Mock converter to return template without placeholder - need to mock update method
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(
                return_value=create_converter_result("No placeholder here")
            )

        # Run setup
        await fuzzer._setup_async(context=basic_fuzzer_context)
        
        # Select a node
        current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        # Should raise MissingPromptPlaceholderException
        with pytest.raises(MissingPromptPlaceholderException, match="Converted template missing placeholder"):
            await fuzzer._apply_template_converter_async(
                context=basic_fuzzer_context,
                current_seed=current_seed
            )

    @pytest.mark.parametrize(
        "jailbreak_count,target_count,expected_outcome",
        [
            (0, 1, AttackOutcome.FAILURE),
            (1, 1, AttackOutcome.SUCCESS),
            (2, 1, AttackOutcome.SUCCESS),
            (1, 2, AttackOutcome.FAILURE),
            (5, 5, AttackOutcome.SUCCESS),
        ],
    )
    @pytest.mark.asyncio
    async def test_attack_outcome_determination(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        jailbreak_count,
        target_count,
        expected_outcome,
    ):
        """Test correct determination of attack outcome based on jailbreak count."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            target_jailbreak_goal_count=target_count,
        )

        # Create context with specific jailbreak count
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={},
            total_jailbreak_count=jailbreak_count,
            jailbreak_conversation_ids=["id"] * jailbreak_count,
            executed_turns=5,
        )

        # Create result
        result = fuzzer._create_attack_result(context)

        # Verify outcome
        assert result.outcome == expected_outcome


@pytest.mark.usefixtures("patch_central_database")
class TestMCTSAlgorithm:
    """Tests for the Monte Carlo Tree Search algorithm implementation."""

    @pytest.mark.parametrize(
        "visited_counts,rewards,expected_index",
        [
            # All unvisited, should select first
            ([0, 0, 0], [0, 0, 0], 0),
            # One visited, should select unvisited
            ([1, 0, 0], [0.5, 0, 0], 1),
            # All visited equally, select highest reward
            ([1, 1, 1], [0.8, 0.5, 0.3], 0),
            # Different visit counts, UCT calculation
            ([5, 2, 1], [2.5, 1.5, 0.8], 2),  # Node 2 has best exploration bonus
        ],
    )
    def test_mcts_node_selection_logic(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        visited_counts,
        rewards,
        expected_index,
    ):
        """Test MCTS node selection follows UCT algorithm."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            frequency_weight=0.5,
        )

        # Create mock nodes with specific states
        nodes = []
        for i in range(len(visited_counts)):
            node = _PromptNode(f"Template {i}")
            node.visited_num = visited_counts[i]
            node.rewards = rewards[i]
            nodes.append(node)

        # Test node selection
        selected_node, path = fuzzer._mcts_explorer.select_node(
            initial_nodes=nodes,
            step=10
        )

        # Verify correct node selected
        assert selected_node == nodes[expected_index]
        assert path[0] == nodes[expected_index]

    def test_mcts_tree_traversal(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test MCTS correctly traverses tree to leaf nodes."""
        # Set random seed for deterministic behavior
        import numpy as np
        np.random.seed(100)
        
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            # Always traverse to leaf
            non_leaf_node_probability=0,  
        )

        # Create tree structure
        root = _PromptNode("Root")
        child1 = _PromptNode("Child1", parent=root) # uct score ~ 0.9
        child2 = _PromptNode("Child2", parent=root) # uct score ~ 1.0
        grandchild = _PromptNode("Grandchild", parent=child2)

        # Set up rewards to make path predictable
        root.visited_num = 3
        root.rewards = 1.0
        child1.visited_num = 2
        child1.rewards = 0.8
        child2.visited_num = 1
        child2.rewards = 0.3
        grandchild.visited_num = 0
        grandchild.rewards = 0

        # Select node - should traverse to grandchild
        selected_node, path = fuzzer._mcts_explorer.select_node(
            initial_nodes=[root],
            step=5
        )

        # Verify path traversal
        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child2
        assert path[2] == grandchild
        assert selected_node == grandchild

    @pytest.mark.parametrize(
        "non_leaf_probability,expected_stop_at_non_leaf",
        [
            (0.0, False), # Never stop at non-leaf
            (1.0, True), # Always stop at non-leaf
        ],
    )
    def test_mcts_non_leaf_node_selection(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        non_leaf_probability,
        expected_stop_at_non_leaf,
    ):
        """Test non-leaf node selection probability."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            non_leaf_node_probability=non_leaf_probability,
        )

        # Create tree with children
        root = _PromptNode("Root")
        child = _PromptNode("Child", parent=root)

        # Mock random to control behavior
        with patch('numpy.random.rand', return_value=0.5):
            selected_node, path = fuzzer._mcts_explorer.select_node(
                initial_nodes=[root],
                step=1
            )

        if expected_stop_at_non_leaf:
            # Should stop at root (non-leaf)
            assert selected_node == root
            assert len(path) == 1
        else:
            # Should continue to child (leaf)
            assert selected_node == child
            assert len(path) == 2

    def test_mcts_reward_updates(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test reward propagation along selected path."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            reward_penalty=0.1,
            minimum_reward=0.2,
        )

        # Create path
        node1 = _PromptNode("Node1")
        node2 = _PromptNode("Node2", parent=node1)
        node3 = _PromptNode("Node3", parent=node2)
        path = [node1, node2, node3]

        # Initial rewards
        initial_rewards = [node.rewards for node in path]

        # Update rewards
        base_reward = 1.0
        fuzzer._mcts_explorer.update_rewards(
            path=path,
            reward=base_reward,
            last_node=node3
        )

        # Verify rewards were updated with decay
        assert node3.rewards > initial_rewards[2]
        assert node2.rewards > initial_rewards[1]
        assert node1.rewards > initial_rewards[0]

        # Node3 is at level 2, so all nodes get: reward = 1.0 * max(0.2, 1 - 0.1 * 2) = 0.8
        assert abs(node3.rewards - 0.8) < 0.001
        assert abs(node2.rewards - 0.8) < 0.001
        assert abs(node1.rewards - 0.8) < 0.001

    def test_mcts_minimum_reward_enforcement(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test minimum reward is enforced during updates."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            reward_penalty=0.5,  # High penalty
            minimum_reward=0.3,
        )

        # Create deep node
        root = _PromptNode("Root")
        current = root
        for i in range(5):  # Create deep tree
            current = _PromptNode(f"Level{i+1}", parent=current)

        path = []
        node = root
        while node:
            path.append(node)
            node = node.children[0] if node.children else None

        # Update rewards
        fuzzer._mcts_explorer.update_rewards(
            path=path,
            reward=1.0,
            last_node=path[-1]
        )

        # Verify minimum reward is enforced for deep nodes
        # Level 5 would have penalty: 1 - 0.5 * 5 = -1.5, but should be clamped to 0.3
        assert path[-1].rewards >= 0.3

    def test_mcts_uct_score_calculation(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test UCT score calculation balances exploration and exploitation."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            frequency_weight=2.0, # High exploration weight
        )

        # Create two nodes with different characteristics
        exploited_node = _PromptNode("Exploited")
        exploited_node.visited_num = 10
        exploited_node.rewards = 5.0 # High average reward (0.5)

        unexplored_node = _PromptNode("Unexplored")
        unexplored_node.visited_num = 1
        unexplored_node.rewards = 0.2 # Low average reward (0.2)

        # At high step count, unexplored node should have higher UCT due to exploration bonus
        step = 100
        exploited_score = fuzzer._mcts_explorer._calculate_uct_score(
            node=exploited_node,
            step=step
        )
        unexplored_score = fuzzer._mcts_explorer._calculate_uct_score(
            node=unexplored_node,
            step=step
        )

        # With high frequency_weight, exploration should dominate
        assert unexplored_score > exploited_score

    def test_mcts_visit_count_updates(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Test visit counts are updated during selection."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Setup context
        basic_fuzzer_context.initial_prompt_nodes = [
            _PromptNode("Template1"),
            _PromptNode("Template2"),
        ]

        # Initial visit counts
        initial_counts = [node.visited_num for node in basic_fuzzer_context.initial_prompt_nodes]

        # Select template
        selected_node, path = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        # Verify visit counts were incremented for path
        for node in path:
            idx = basic_fuzzer_context.initial_prompt_nodes.index(node) if node in basic_fuzzer_context.initial_prompt_nodes else -1
            if idx >= 0:
                assert node.visited_num == initial_counts[idx] + 1

        # Verify executed turns incremented
        assert basic_fuzzer_context.executed_turns == 1

    @pytest.mark.asyncio
    async def test_mcts_integration_with_attack_flow(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_prompt_response,
        create_score,
        create_converter_result,
    ):
        """Test MCTS integration in full attack iteration."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            frequency_weight=0.5,
            reward_penalty=0.1,
        )

        # Mock converters with update method
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(
                return_value=create_converter_result("Converted: {{ prompt }}")
            )

        # Mock responses and scores
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
            return_value=responses
        )

        # Mock high scores for jailbreak
        high_scores = [create_score(0.9) for _ in range(len(sample_prompts))]
        attack_scoring_config.objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(
            return_value=high_scores
        )

        # Setup
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Capture initial state
        initial_node = basic_fuzzer_context.initial_prompt_nodes[0]
        initial_rewards = initial_node.rewards
        initial_visits = initial_node.visited_num

        # Execute iteration
        await fuzzer._execute_attack_iteration_async(basic_fuzzer_context)

        # Verify MCTS state was updated
        assert initial_node.visited_num > initial_visits
        assert initial_node.rewards > initial_rewards
        assert len(basic_fuzzer_context.mcts_selected_path) > 0
        assert basic_fuzzer_context.last_choice_node is not None

    def test_mcts_handles_empty_tree(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test MCTS handles edge case of empty initial nodes."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # This should raise an error as we need at least one node
        with pytest.raises(ValueError):
            fuzzer._mcts_explorer.select_node(
                initial_nodes=[],
                step=1
            )

    def test_mcts_deterministic_selection(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test MCTS selection is deterministic given same state."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            non_leaf_node_probability=0,  # Remove randomness
        )

        # Create identical node states
        nodes1 = [_PromptNode(f"Template{i}") for i in range(3)]
        nodes2 = [_PromptNode(f"Template{i}") for i in range(3)]

        # Set same state
        for i in range(3):
            nodes1[i].visited_num = i
            nodes1[i].rewards = i * 0.5
            nodes2[i].visited_num = i
            nodes2[i].rewards = i * 0.5

        # Select from both sets
        selected1, _ = fuzzer._mcts_explorer.select_node(initial_nodes=nodes1, step=10)
        selected2, _ = fuzzer._mcts_explorer.select_node(initial_nodes=nodes2, step=10)

        # Should select same index
        idx1 = nodes1.index(selected1)
        idx2 = nodes2.index(selected2)
        assert idx1 == idx2
