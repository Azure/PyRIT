# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.attacks.base.attack_config import AttackConverterConfig, AttackScoringConfig
from pyrit.attacks.fuzzer import (
    FuzzerAttack,
    FuzzerAttackContext,
    FuzzerAttackResult,
    _MCTSExplorer,
    _PromptNode,
)
from pyrit.exceptions import (
    AttackValidationException,
    MissingPromptPlaceholderException,
)
from pyrit.models import (
    AttackOutcome,
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    SeedPrompt,
)
from pyrit.prompt_converter import (
    ConverterResult,
    FuzzerCrossOverConverter,
    FuzzerExpandConverter,
    FuzzerRephraseConverter,
    FuzzerShortenConverter,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import FloatScaleThresholdScorer


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

        with pytest.raises(MissingPromptPlaceholderException, match="Template missing placeholder"):
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
            (-0.1, True),  # No validation on frequency_weight
            (1.1, True),  # No validation on frequency_weight
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
        with patch("pyrit.attacks.fuzzer.SelfAskScaleScorer") as mock_self_ask_scorer_class:
            with patch("pyrit.attacks.fuzzer.FloatScaleThresholdScorer") as mock_threshold_scorer_class:
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
                assert call_args.kwargs["chat_target"] == mock_scoring_target

                # Check that the enum paths are being accessed correctly
                assert "scale_arguments_path" in call_args.kwargs
                assert "system_prompt_path" in call_args.kwargs

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
        with patch("pyrit.attacks.fuzzer.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer.FloatScaleThresholdScorer"):
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
        with patch("pyrit.attacks.fuzzer.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer.FloatScaleThresholdScorer"):
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
        with patch("pyrit.attacks.fuzzer.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer.FloatScaleThresholdScorer"):
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
        with patch("pyrit.attacks.fuzzer.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer.FloatScaleThresholdScorer"):
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
        with patch("pyrit.attacks.fuzzer.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer.FloatScaleThresholdScorer"):
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
        with patch("pyrit.attacks.fuzzer.SelfAskScaleScorer"):
            with patch("pyrit.attacks.fuzzer.FloatScaleThresholdScorer"):
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
            ("A" * 1000, True),  # Long objective
            ("123", True),  # Numeric string
            ("", False),  # Empty string
            ("   ", True),  # Whitespace (passes validation but may not be meaningful)
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
            attack_input=objective,
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

        with patch.object(
            fuzzer, "_validate_context", side_effect=ValueError("The attack objective must be set in the context.")
        ) as mock_validate:
            # Create invalid context
            context = FuzzerAttackContext(
                objective="",  # Invalid
                memory_labels={},
            )

            # Try to execute attack - should call validate_context
            import asyncio

            with pytest.raises(
                AttackValidationException,
                match="Context validation failed: The attack objective must be set in the context",
            ):
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
            assert node.level == 0  # Root nodes
            assert node.parent is None
            assert node.visited_num == 0
            assert node.rewards == 0
            assert node.children == []
            assert node.id is not None  # UUID should be assigned

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
            "shared_key": "context_value",  # Context value takes precedence
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
            converter.convert_async = AsyncMock(return_value=create_converter_result("Converted: {{ prompt }}"))

        # Mock normalizer to return responses
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(return_value=responses)

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
            converter.convert_async = AsyncMock(return_value=create_converter_result("Converted: {{ prompt }}"))

        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(return_value=responses)

        # Mock scorer to return low scores (no jailbreak)
        low_scores = [create_score(0.3) for _ in range(len(sample_prompts))]
        attack_scoring_config.objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(return_value=low_scores)

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
            converter.convert_async = AsyncMock(return_value=create_converter_result("Converted: {{ prompt }}"))

        # Mock normalizer
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(return_value=responses)

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
            converter.convert_async = AsyncMock(side_effect=MissingPromptPlaceholderException())

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
            converter.convert_async = AsyncMock(return_value=create_converter_result("Converted: {{ prompt }}"))

        # Mock normalizer
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(return_value=responses)

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
        await fuzzer._send_prompts_to_target_async(context=basic_fuzzer_context, prompts=many_prompts)

        # Verify batch_size was passed to normalizer
        call_args = fuzzer._prompt_normalizer.send_prompt_batch_to_target_async.call_args
        assert call_args.kwargs["batch_size"] == batch_size

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
            converter.convert_async = AsyncMock(return_value=create_converter_result("No placeholder here"))

        # Run setup
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Select a node
        current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        # Should raise MissingPromptPlaceholderException
        with pytest.raises(MissingPromptPlaceholderException, match="Converted template missing placeholder"):
            await fuzzer._apply_template_converter_async(context=basic_fuzzer_context, current_seed=current_seed)

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
        selected_node, path = fuzzer._mcts_explorer.select_node(initial_nodes=nodes, step=10)

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
        child1 = _PromptNode("Child1", parent=root)  # uct score ~ 0.9
        child2 = _PromptNode("Child2", parent=root)  # uct score ~ 1.0
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
        selected_node, path = fuzzer._mcts_explorer.select_node(initial_nodes=[root], step=5)

        # Verify path traversal
        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child2
        assert path[2] == grandchild
        assert selected_node == grandchild

    @pytest.mark.parametrize(
        "non_leaf_probability,expected_stop_at_non_leaf",
        [
            (0.0, False),  # Never stop at non-leaf
            (1.0, True),  # Always stop at non-leaf
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
        with patch("numpy.random.rand", return_value=0.5):
            selected_node, path = fuzzer._mcts_explorer.select_node(initial_nodes=[root], step=1)

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
        fuzzer._mcts_explorer.update_rewards(path=path, reward=base_reward, last_node=node3)

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
        fuzzer._mcts_explorer.update_rewards(path=path, reward=1.0, last_node=path[-1])

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
            frequency_weight=2.0,  # High exploration weight
        )

        # Create two nodes with different characteristics
        exploited_node = _PromptNode("Exploited")
        exploited_node.visited_num = 10
        exploited_node.rewards = 5.0  # High average reward (0.5)

        unexplored_node = _PromptNode("Unexplored")
        unexplored_node.visited_num = 1
        unexplored_node.rewards = 0.2  # Low average reward (0.2)

        # At high step count, unexplored node should have higher UCT due to exploration bonus
        step = 100
        exploited_score = fuzzer._mcts_explorer._calculate_uct_score(node=exploited_node, step=step)
        unexplored_score = fuzzer._mcts_explorer._calculate_uct_score(node=unexplored_node, step=step)

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
            idx = (
                basic_fuzzer_context.initial_prompt_nodes.index(node)
                if node in basic_fuzzer_context.initial_prompt_nodes
                else -1
            )
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
            converter.convert_async = AsyncMock(return_value=create_converter_result("Converted: {{ prompt }}"))

        # Mock responses and scores
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(return_value=responses)

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
            fuzzer._mcts_explorer.select_node(initial_nodes=[], step=1)

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


@pytest.mark.usefixtures("patch_central_database")
class TestTemplateConverters:
    """Tests template converter integration."""

    @pytest.mark.asyncio
    async def test_apply_template_converter_single(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
        create_converter_result,
    ):
        """Test with single converter."""
        # Create a single converter
        single_converter = FuzzerExpandConverter(converter_target=mock_scoring_target)
        single_converter.update = MagicMock()
        single_converter.convert_async = AsyncMock(return_value=create_converter_result("Expanded: {{ prompt }}"))

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=[single_converter],
            attack_scoring_config=attack_scoring_config,
        )

        # Setup context
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Select a node
        current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        # Apply converter
        result = await fuzzer._apply_template_converter_async(context=basic_fuzzer_context, current_seed=current_seed)

        # Verify converter was called
        single_converter.convert_async.assert_called_once_with(prompt=current_seed.template)
        assert result == "Expanded: {{ prompt }}"
        assert "{{ prompt }}" in result

    @pytest.mark.asyncio
    async def test_apply_template_converter_multiple(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
        create_converter_result,
    ):
        """Test random selection from multiple converters."""
        # Create multiple converters
        converters = []
        for i, converter_type in enumerate([FuzzerExpandConverter, FuzzerShortenConverter, FuzzerRephraseConverter]):
            converter = converter_type(converter_target=mock_scoring_target)
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(return_value=create_converter_result(f"Converted{i}: {{{{ prompt }}}}"))
            converters.append(converter)

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Setup context
        await fuzzer._setup_async(context=basic_fuzzer_context)
        current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        # Mock random.choice to control which converter is selected
        with patch("random.choice", return_value=converters[1]):
            result = await fuzzer._apply_template_converter_async(
                context=basic_fuzzer_context, current_seed=current_seed
            )

        # Verify only the selected converter was called
        converters[1].convert_async.assert_called_once()
        converters[0].convert_async.assert_not_called()
        converters[2].convert_async.assert_not_called()
        assert result == "Converted1: {{ prompt }}"

    @pytest.mark.asyncio
    async def test_apply_template_converter_with_other_templates(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
        create_converter_result,
    ):
        """Test CrossOverConverter with other templates."""
        # Create crossover converter
        crossover_converter = FuzzerCrossOverConverter(converter_target=mock_scoring_target)
        crossover_converter.update = MagicMock()
        crossover_converter.convert_async = AsyncMock(return_value=create_converter_result("CrossOver: {{ prompt }}"))

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=[crossover_converter],
            attack_scoring_config=attack_scoring_config,
        )

        # Setup context with additional nodes
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Add some successful nodes
        for i in range(2):
            new_node = _PromptNode(f"Success template {i}: {{{{ prompt }}}}")
            basic_fuzzer_context.new_prompt_nodes.append(new_node)

        current_seed, path = fuzzer._select_template_with_mcts(basic_fuzzer_context)
        basic_fuzzer_context.mcts_selected_path = path

        # Apply converter
        result = await fuzzer._apply_template_converter_async(context=basic_fuzzer_context, current_seed=current_seed)

        # Verify update was called with other templates
        crossover_converter.update.assert_called_once()
        update_args = crossover_converter.update.call_args
        other_templates = update_args.kwargs["prompt_templates"]

        # Should include initial templates not in path and new successful templates
        assert len(other_templates) > 0
        # Should not include the current seed template
        assert current_seed.template not in other_templates

        assert result == "CrossOver: {{ prompt }}"

    @pytest.mark.asyncio
    async def test_apply_template_converter_preserves_placeholder(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
        create_converter_result,
    ):
        """Verify {{ prompt }} placeholder is preserved."""
        # Test various converter outputs
        test_cases = [
            "Before {{ prompt }} after",
            "{{ prompt }} at start",
            "At end {{ prompt }}",
            "Multiple {{ prompt }} placeholders {{ prompt }}",
        ]

        for test_template in test_cases:
            converter = FuzzerRephraseConverter(converter_target=mock_scoring_target)
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(return_value=create_converter_result(test_template))

            fuzzer = FuzzerAttack(
                objective_target=mock_objective_target,
                prompts=sample_prompts,
                prompt_templates=sample_templates,
                template_converters=[converter],
                attack_scoring_config=attack_scoring_config,
            )

            await fuzzer._setup_async(context=basic_fuzzer_context)
            current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

            result = await fuzzer._apply_template_converter_async(
                context=basic_fuzzer_context, current_seed=current_seed
            )

            assert "{{ prompt }}" in result
            assert result == test_template

    @pytest.mark.asyncio
    async def test_apply_template_converter_retry_on_missing_placeholder(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
        create_converter_result,
    ):
        """Test retry mechanism when placeholder is lost."""
        converter = FuzzerExpandConverter(converter_target=mock_scoring_target)
        converter.update = MagicMock()

        # First call returns template without placeholder, second call returns with placeholder
        converter.convert_async = AsyncMock(
            side_effect=[
                create_converter_result("No placeholder here"),
                create_converter_result("With {{ prompt }} placeholder"),
            ]
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=[converter],
            attack_scoring_config=attack_scoring_config,
        )

        await fuzzer._setup_async(context=basic_fuzzer_context)
        current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        # Apply converter - should retry and succeed
        result = await fuzzer._apply_template_converter_async(context=basic_fuzzer_context, current_seed=current_seed)

        # Verify converter was called twice
        assert converter.convert_async.call_count == 2
        assert result == "With {{ prompt }} placeholder"

    @pytest.mark.asyncio
    async def test_apply_template_converter_max_retries_exceeded(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
        create_converter_result,
    ):
        """Test when all retry attempts fail to produce valid template."""
        converter = FuzzerExpandConverter(converter_target=mock_scoring_target)
        converter.update = MagicMock()

        # Always return template without placeholder
        converter.convert_async = AsyncMock(return_value=create_converter_result("No placeholder"))

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=[converter],
            attack_scoring_config=attack_scoring_config,
        )

        await fuzzer._setup_async(context=basic_fuzzer_context)
        current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        # Should raise after max retries
        with pytest.raises(MissingPromptPlaceholderException, match="Converted template missing placeholder"):
            await fuzzer._apply_template_converter_async(context=basic_fuzzer_context, current_seed=current_seed)

    @pytest.mark.asyncio
    async def test_get_other_templates_excludes_path(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Test _get_other_templates excludes templates in current MCTS path."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Select a path
        selected_node, path = fuzzer._select_template_with_mcts(basic_fuzzer_context)
        basic_fuzzer_context.mcts_selected_path = path

        # Add some new nodes
        new_nodes = [
            _PromptNode("New template 1: {{ prompt }}"),
            _PromptNode("New template 2: {{ prompt }}"),
        ]
        basic_fuzzer_context.new_prompt_nodes.extend(new_nodes)

        # Get other templates
        other_templates = fuzzer._get_other_templates(basic_fuzzer_context)

        # Verify path templates are excluded
        path_templates = {node.template for node in path}
        for template in other_templates:
            assert template not in path_templates

        # Verify includes initial templates not in path and new templates
        expected_count = len(sample_templates) - len(path) + len(new_nodes)
        assert len(other_templates) == expected_count

    @pytest.mark.parametrize(
        "converter_type,expected_behavior",
        [
            (FuzzerExpandConverter, "expands"),
            (FuzzerShortenConverter, "shortens"),
            (FuzzerRephraseConverter, "rephrases"),
            (FuzzerCrossOverConverter, "crosses over"),
        ],
    )
    @pytest.mark.asyncio
    async def test_converter_types_behavior(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
        create_converter_result,
        converter_type,
        expected_behavior,
    ):
        """Test different converter types are properly integrated."""
        converter = converter_type(converter_target=mock_scoring_target)
        converter.update = MagicMock()
        converter.convert_async = AsyncMock(
            return_value=create_converter_result(f"{expected_behavior}: {{{{ prompt }}}}")
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=[converter],
            attack_scoring_config=attack_scoring_config,
        )

        await fuzzer._setup_async(context=basic_fuzzer_context)
        current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        result = await fuzzer._apply_template_converter_async(context=basic_fuzzer_context, current_seed=current_seed)

        # Verify converter was called and result contains expected behavior
        converter.convert_async.assert_called_once()
        assert expected_behavior in result
        assert "{{ prompt }}" in result

    @pytest.mark.asyncio
    async def test_converter_update_called_before_convert(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
        create_converter_result,
    ):
        """Test that converter.update() is called before converter.convert_async()."""
        converter = FuzzerCrossOverConverter(converter_target=mock_scoring_target)

        call_order = []
        converter.update = MagicMock(side_effect=lambda **kwargs: call_order.append("update"))
        converter.convert_async = AsyncMock(
            side_effect=lambda **kwargs: call_order.append("convert") or create_converter_result("Result: {{ prompt }}")
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=[converter],
            attack_scoring_config=attack_scoring_config,
        )

        await fuzzer._setup_async(context=basic_fuzzer_context)
        current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        await fuzzer._apply_template_converter_async(context=basic_fuzzer_context, current_seed=current_seed)

        # Verify update was called before convert
        assert call_order == ["update", "convert"]
        converter.update.assert_called_once()
        converter.convert_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_converter_error_propagation(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
    ):
        """Test that converter errors are properly propagated."""
        converter = FuzzerExpandConverter(converter_target=mock_scoring_target)
        converter.update = MagicMock()

        # Simulate converter error
        converter.convert_async = AsyncMock(side_effect=Exception("Converter failed"))

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=[converter],
            attack_scoring_config=attack_scoring_config,
        )

        await fuzzer._setup_async(context=basic_fuzzer_context)
        current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        # Should propagate the exception
        with pytest.raises(Exception, match="Converter failed"):
            await fuzzer._apply_template_converter_async(context=basic_fuzzer_context, current_seed=current_seed)

    @pytest.mark.asyncio
    async def test_empty_other_templates_handling(
        self,
        mock_objective_target,
        sample_prompts,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
        create_converter_result,
    ):
        """Test converter behavior when no other templates are available."""
        # Single template scenario
        single_template = ["Only template: {{ prompt }}"]

        converter = FuzzerCrossOverConverter(converter_target=mock_scoring_target)
        converter.update = MagicMock()
        converter.convert_async = AsyncMock(return_value=create_converter_result("CrossOver: {{ prompt }}"))

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=single_template,
            template_converters=[converter],
            attack_scoring_config=attack_scoring_config,
        )

        await fuzzer._setup_async(context=basic_fuzzer_context)
        current_seed, path = fuzzer._select_template_with_mcts(basic_fuzzer_context)
        basic_fuzzer_context.mcts_selected_path = path

        # Apply converter with no other templates available
        result = await fuzzer._apply_template_converter_async(context=basic_fuzzer_context, current_seed=current_seed)

        # Verify update was called with empty list
        converter.update.assert_called_once_with(prompt_templates=[])
        assert result == "CrossOver: {{ prompt }}"


@pytest.mark.usefixtures("patch_central_database")
class TestPromptNodeTree:
    """Tests the tree structure management."""

    def test_prompt_node_initialization(self):
        """Test node creation with/without parent."""
        # Test node without parent
        node = _PromptNode("Test template: {{ prompt }}")

        assert node.template == "Test template: {{ prompt }}"
        assert node.parent is None
        assert node.level == 0
        assert node.visited_num == 0
        assert node.rewards == 0
        assert node.children == []
        assert node.id is not None
        assert isinstance(node.id, uuid.UUID)

        # Test node with parent
        parent_node = _PromptNode("Parent template: {{ prompt }}")
        child_node = _PromptNode("Child template: {{ prompt }}", parent=parent_node)

        assert child_node.parent == parent_node
        assert child_node.level == 1
        assert child_node in parent_node.children
        assert len(parent_node.children) == 1

    def test_prompt_node_add_parent(self):
        """Test parent-child relationships and level calculation."""
        # Create nodes independently
        root = _PromptNode("Root: {{ prompt }}")
        child1 = _PromptNode("Child1: {{ prompt }}")
        child2 = _PromptNode("Child2: {{ prompt }}")
        grandchild = _PromptNode("Grandchild: {{ prompt }}")

        # Verify initial state
        assert root.level == 0
        assert child1.level == 0
        assert child2.level == 0
        assert grandchild.level == 0

        # Build tree structure
        child1.add_parent(root)
        child2.add_parent(root)
        grandchild.add_parent(child1)

        # Verify parent-child relationships
        assert child1.parent == root
        assert child2.parent == root
        assert grandchild.parent == child1

        # Verify children lists
        assert len(root.children) == 2
        assert child1 in root.children
        assert child2 in root.children
        assert len(child1.children) == 1
        assert grandchild in child1.children
        assert len(child2.children) == 0

        # Verify level calculation
        assert root.level == 0
        assert child1.level == 1
        assert child2.level == 1
        assert grandchild.level == 2

    def test_prompt_node_tree_traversal(self):
        """Test path selection in tree."""
        # Build a tree
        root = _PromptNode("Root: {{ prompt }}")
        child1 = _PromptNode("Child1: {{ prompt }}", parent=root)
        child2 = _PromptNode("Child2: {{ prompt }}", parent=root)
        grandchild1 = _PromptNode("Grandchild1: {{ prompt }}", parent=child1)
        grandchild2 = _PromptNode("Grandchild2: {{ prompt }}", parent=child1)
        grandchild3 = _PromptNode("Grandchild3: {{ prompt }}", parent=child2)

        # Test traversal from root to leaf
        def get_path_to_node(target_node: _PromptNode) -> List[_PromptNode]:
            path = []
            current = target_node
            while current is not None:
                path.append(current)
                current = current.parent
            return list(reversed(path))

        # Test various paths
        path_to_grandchild1 = get_path_to_node(grandchild1)
        assert path_to_grandchild1 == [root, child1, grandchild1]

        path_to_grandchild3 = get_path_to_node(grandchild3)
        assert path_to_grandchild3 == [root, child2, grandchild3]

        path_to_child2 = get_path_to_node(child2)
        assert path_to_child2 == [root, child2]

        # Test finding all leaf nodes
        def get_leaf_nodes(node: _PromptNode) -> List[_PromptNode]:
            if not node.children:
                return [node]

            leaves = []
            for child in node.children:
                leaves.extend(get_leaf_nodes(child))
            return leaves

        leaf_nodes = get_leaf_nodes(root)
        assert len(leaf_nodes) == 3
        assert grandchild1 in leaf_nodes
        assert grandchild2 in leaf_nodes
        assert grandchild3 in leaf_nodes

    @pytest.mark.asyncio
    async def test_add_successful_node_to_tree(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Test adding new nodes when jailbreak succeeds."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Setup context
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Select a parent node
        parent_node = basic_fuzzer_context.initial_prompt_nodes[0]

        # Create successful template node
        successful_template = "Successful: {{ prompt }}"
        template_node = _PromptNode(successful_template)

        # Add successful template
        fuzzer._add_successful_template(
            context=basic_fuzzer_context, template_node=template_node, parent_seed=parent_node
        )

        # Verify node was added
        assert len(basic_fuzzer_context.new_prompt_nodes) == 1
        assert basic_fuzzer_context.new_prompt_nodes[0] == template_node
        assert template_node.parent == parent_node
        assert template_node in parent_node.children
        assert template_node.level == parent_node.level + 1

        # Test duplicate prevention
        duplicate_node = _PromptNode(successful_template)
        initial_count = len(basic_fuzzer_context.new_prompt_nodes)

        fuzzer._add_successful_template(
            context=basic_fuzzer_context, template_node=duplicate_node, parent_seed=parent_node
        )

        # Should not add duplicate
        assert len(basic_fuzzer_context.new_prompt_nodes) == initial_count

    def test_prompt_node_identity(self):
        """Test node identity and uniqueness."""
        # Create multiple nodes with same template
        template = "Same template: {{ prompt }}"
        node1 = _PromptNode(template)
        node2 = _PromptNode(template)
        node3 = _PromptNode(template)

        # Verify unique IDs
        assert node1.id != node2.id
        assert node1.id != node3.id
        assert node2.id != node3.id

        # Verify template equality
        assert node1.template == node2.template == node3.template

        # Test node comparison
        assert node1 != node2  # Different objects
        assert node1.id == node1.id  # Same ID

    def test_prompt_node_state_updates(self):
        """Test updating node state (visits, rewards)."""
        node = _PromptNode("Test: {{ prompt }}")

        # Initial state
        assert node.visited_num == 0
        assert node.rewards == 0

        # Update visits
        node.visited_num += 1
        assert node.visited_num == 1

        node.visited_num += 5
        assert node.visited_num == 6

        # Update rewards
        node.rewards += 0.5
        assert node.rewards == 0.5

        node.rewards += 0.3
        assert abs(node.rewards - 0.8) < 0.001

    @pytest.mark.parametrize(
        "tree_depth,expected_levels",
        [
            (1, [0]),
            (2, [0, 1]),
            (3, [0, 1, 2]),
            (5, [0, 1, 2, 3, 4]),
        ],
    )
    def test_prompt_node_deep_tree(self, tree_depth, expected_levels):
        """Test deep tree structures."""
        nodes = []
        current = _PromptNode("Level 0: {{{{ prompt }}}}")
        nodes.append(current)

        # Build deep tree
        for i in range(1, tree_depth):
            child = _PromptNode(f"Level {i}: {{{{ prompt }}}}", parent=current)
            nodes.append(child)
            current = child

        # Verify levels
        for i, node in enumerate(nodes):
            assert node.level == expected_levels[i]

        # Verify parent-child chain
        for i in range(len(nodes) - 1):
            assert nodes[i + 1].parent == nodes[i]
            assert nodes[i + 1] in nodes[i].children

    def test_prompt_node_multi_child_tree(self):
        """Test tree with multiple children per node."""
        root = _PromptNode("Root: {{ prompt }}")

        # Add multiple children to root
        children = []
        for i in range(5):
            child = _PromptNode(f"Child {i}: {{{{ prompt }}}}", parent=root)
            children.append(child)

        assert len(root.children) == 5
        for i, child in enumerate(children):
            assert child in root.children
            assert child.parent == root
            assert child.level == 1

        # Add grandchildren to first child
        grandchildren = []
        for i in range(3):
            grandchild = _PromptNode(f"Grandchild {i}: {{{{ prompt }}}}", parent=children[0])
            grandchildren.append(grandchild)

        assert len(children[0].children) == 3
        for grandchild in grandchildren:
            assert grandchild.parent == children[0]
            assert grandchild.level == 2

    def test_get_other_templates_logic(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
    ):
        """Test _get_other_templates excludes path nodes correctly."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Create mock nodes
        node1 = _PromptNode("Template 1: {{ prompt }}")
        node2 = _PromptNode("Template 2: {{ prompt }}")
        node3 = _PromptNode("Template 3: {{ prompt }}")
        node4 = _PromptNode("Template 4: {{ prompt }}")

        # Set up context
        basic_fuzzer_context.initial_prompt_nodes = [node1, node2, node3]
        basic_fuzzer_context.new_prompt_nodes = [node4]
        basic_fuzzer_context.mcts_selected_path = [node1, node3]  # Exclude these

        # Get other templates
        other_templates = fuzzer._get_other_templates(basic_fuzzer_context)

        # Should include node2 and node4, but not node1 or node3
        assert len(other_templates) == 2
        assert "Template 2: {{ prompt }}" in other_templates
        assert "Template 4: {{ prompt }}" in other_templates
        assert "Template 1: {{ prompt }}" not in other_templates
        assert "Template 3: {{ prompt }}" not in other_templates

    def test_prompt_node_isolation(self):
        """Test that modifying one node doesn't affect others."""
        # Create sibling nodes
        parent = _PromptNode("Parent: {{ prompt }}")
        child1 = _PromptNode("Child 1: {{ prompt }}", parent=parent)
        child2 = _PromptNode("Child 2: {{ prompt }}", parent=parent)

        # Modify child1
        child1.visited_num = 10
        child1.rewards = 5.0
        grandchild = _PromptNode("Grandchild: {{ prompt }}", parent=child1)

        # Verify child2 is unaffected
        assert child2.visited_num == 0
        assert child2.rewards == 0
        assert len(child2.children) == 0

        # Verify child1 has the grandchild
        assert len(child1.children) == 1
        assert grandchild in child1.children

        # Verify parent only shows connection
        assert len(parent.children) == 2
        assert parent.visited_num == 0
        assert parent.rewards == 0


@pytest.mark.usefixtures("patch_central_database")
class TestScoringIntegration:
    """Tests scoring and jailbreak detection."""

    @pytest.mark.asyncio
    async def test_score_responses_with_objective_scorer(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_objective_scorer,
        basic_fuzzer_context,
        create_prompt_response,
        create_score,
    ):
        """Mock scorer responses and verify score interpretation."""
        # Configure scoring
        scoring_config = AttackScoringConfig(
            objective_scorer=mock_objective_scorer,
            successful_objective_threshold=0.8,
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=scoring_config,
        )

        # Create test responses
        responses = [create_prompt_response(f"Response {i}") for i in range(3)]

        # Mock scorer to return different scores
        mock_scores = [
            create_score(0.9),  # Jailbreak
            create_score(0.5),  # Not jailbreak
            create_score(0.85),  # Jailbreak
        ]
        mock_objective_scorer.score_prompts_with_tasks_batch_async.return_value = mock_scores

        # Score responses
        scores = await fuzzer._score_responses_async(responses=responses)

        # Verify scorer was called correctly
        mock_objective_scorer.score_prompts_with_tasks_batch_async.assert_called_once()
        call_args = mock_objective_scorer.score_prompts_with_tasks_batch_async.call_args

        # Verify request pieces were extracted
        request_pieces = call_args.kwargs["request_responses"]
        assert len(request_pieces) == 3
        for i, piece in enumerate(request_pieces):
            assert piece == responses[i].request_pieces[0]

        # Verify tasks
        assert call_args.kwargs["tasks"] == sample_prompts

        # Verify scores returned
        assert scores == mock_scores

    def test_identify_successful_jailbreaks(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        create_score,
    ):
        """Test threshold comparison (0.8 default)."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Test various score values
        test_cases = [
            (create_score(0.9), True),  # Above threshold
            (create_score(0.8), True),  # At threshold
            (create_score(0.79), False),  # Just below threshold
            (create_score(0.5), False),  # Well below threshold
            (create_score(1.0), True),  # Maximum score
            (create_score(0.0), False),  # Minimum score
        ]

        for score, expected_jailbreak in test_cases:
            is_jailbreak = fuzzer._is_jailbreak(score)
            assert (
                is_jailbreak == expected_jailbreak
            ), f"Score {score.get_value()} should {'be' if expected_jailbreak else 'not be'} a jailbreak"

    @pytest.mark.asyncio
    async def test_track_jailbreak_conversation_ids(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_prompt_response,
        create_score,
    ):
        """Verify conversation IDs are tracked for successful jailbreaks."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Create responses with unique conversation IDs
        conversation_ids = [f"conv-{i}" for i in range(3)]
        responses = [create_prompt_response(f"Response {i}", conversation_id=conversation_ids[i]) for i in range(3)]

        # Create scores (first and third are jailbreaks)
        scores = [
            create_score(0.9),  # Jailbreak
            create_score(0.5),  # Not jailbreak
            create_score(0.85),  # Jailbreak
        ]

        # Create template node
        template_node = _PromptNode("Test: {{ prompt }}")
        current_seed = (
            basic_fuzzer_context.initial_prompt_nodes[0]
            if basic_fuzzer_context.initial_prompt_nodes
            else _PromptNode("Seed")
        )

        # Process scoring results
        jailbreak_count = fuzzer._process_scoring_results(
            context=basic_fuzzer_context,
            scores=scores,
            responses=responses,
            template_node=template_node,
            current_seed=current_seed,
        )

        # Verify jailbreak tracking
        assert jailbreak_count == 2
        assert basic_fuzzer_context.total_jailbreak_count == 2
        assert len(basic_fuzzer_context.jailbreak_conversation_ids) == 2
        assert conversation_ids[0] in basic_fuzzer_context.jailbreak_conversation_ids
        assert conversation_ids[1] not in basic_fuzzer_context.jailbreak_conversation_ids
        assert conversation_ids[2] in basic_fuzzer_context.jailbreak_conversation_ids

        # Verify query count tracking
        assert basic_fuzzer_context.total_target_query_count == 3

    def test_no_scorer_configuration_error(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
    ):
        """Test behavior when no scorer is configured."""
        # Create scoring config without objective scorer
        scoring_config = AttackScoringConfig(
            objective_scorer=None,
            auxiliary_scorers=[],
        )

        # Should raise error during initialization
        with pytest.raises(ValueError, match="Objective scorer must be provided"):
            FuzzerAttack(
                objective_target=mock_objective_target,
                prompts=sample_prompts,
                prompt_templates=sample_templates,
                template_converters=sample_converters,
                attack_scoring_config=scoring_config,
            )

    @pytest.mark.asyncio
    async def test_auxiliary_scorers_integration(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_objective_scorer,
        mock_auxiliary_scorer,
        create_prompt_response,
        create_score,
    ):
        """Test that auxiliary scorers are called during scoring."""
        # Create additional auxiliary scorer
        auxiliary_scorer2 = MagicMock()
        auxiliary_scorer2.score_prompts_with_tasks_batch_async = AsyncMock()

        scoring_config = AttackScoringConfig(
            objective_scorer=mock_objective_scorer,
            auxiliary_scorers=[mock_auxiliary_scorer, auxiliary_scorer2],
            successful_objective_threshold=0.8,
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=scoring_config,
        )

        # Create responses
        responses = [create_prompt_response(f"Response {i}") for i in range(2)]

        # Mock objective scorer
        mock_objective_scorer.score_prompts_with_tasks_batch_async.return_value = [
            create_score(0.9),
            create_score(0.7),
        ]

        # Mock auxiliary scorers to return values (even though they are not used in the return)
        mock_auxiliary_scorer.score_prompts_with_tasks_batch_async.return_value = [
            create_score(0.8),
            create_score(0.6),
        ]
        auxiliary_scorer2.score_prompts_with_tasks_batch_async.return_value = [
            create_score(0.7),
            create_score(0.5),
        ]

        # Score responses
        scores = await fuzzer._score_responses_async(responses=responses)

        # Verify all scorers were called
        mock_objective_scorer.score_prompts_with_tasks_batch_async.assert_called_once()
        mock_auxiliary_scorer.score_prompts_with_tasks_batch_async.assert_called_once()
        auxiliary_scorer2.score_prompts_with_tasks_batch_async.assert_called_once()

        # Verify they received the same arguments
        for scorer in [mock_objective_scorer, mock_auxiliary_scorer, auxiliary_scorer2]:
            call_args = scorer.score_prompts_with_tasks_batch_async.call_args
            assert len(call_args.kwargs["request_responses"]) == 2
            assert call_args.kwargs["tasks"] == sample_prompts

        # Verify that only objective scores are returned
        assert len(scores) == 2
        assert all(score.get_value() in [0.9, 0.7] for score in scores)

    @pytest.mark.parametrize(
        "score_value,expected_normalized",
        [
            (True, 1.0),
            (False, 0.0),
            (1, 1.0),
            (0, 0.0),
            (0.5, 0.5),
            (0.75, 0.75),
            (1.5, 1.0),  # Clamped to max
            (-0.5, 0.0),  # Clamped to min
            (2.0, 1.0),  # Clamped to max
            (-10, 0.0),  # Clamped to min
            ("invalid", 0.0),  # Invalid type
            (None, 0.0),  # None value
        ],
    )
    def test_normalize_score_to_float(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        score_value,
        expected_normalized,
    ):
        """Test score normalization for different score types."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        normalized = fuzzer._normalize_score_to_float(score_value)
        assert normalized == expected_normalized

    @pytest.mark.asyncio
    async def test_empty_responses_scoring(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test scoring with empty response list."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Score empty responses
        scores = await fuzzer._score_responses_async(responses=[])

        # Should return empty list
        assert scores == []

        # Scorer should not be called
        attack_scoring_config.objective_scorer.score_prompts_with_tasks_batch_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_scoring_error_propagation(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_objective_scorer,
        basic_fuzzer_context,
        create_prompt_response,
    ):
        """Test that scoring errors are properly propagated."""
        scoring_config = AttackScoringConfig(
            objective_scorer=mock_objective_scorer,
            successful_objective_threshold=0.8,
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=scoring_config,
        )

        # Mock scorer to raise error
        mock_objective_scorer.score_prompts_with_tasks_batch_async.side_effect = RuntimeError("Scoring failed")

        # Create responses
        responses = [create_prompt_response("Test response")]

        # Should propagate the error
        with pytest.raises(RuntimeError, match="Scoring failed"):
            await fuzzer._score_responses_async(responses=responses)

    def test_successful_template_tracking(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_prompt_response,
        create_score,
    ):
        """Test that successful templates are properly tracked."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Setup initial nodes
        basic_fuzzer_context.initial_prompt_nodes = [_PromptNode(template) for template in sample_templates]
        parent_seed = basic_fuzzer_context.initial_prompt_nodes[0]

        # Create responses and scores
        responses = [create_prompt_response(f"Response {i}") for i in range(3)]

        # Test with no jailbreaks
        low_scores = [create_score(0.3) for _ in range(3)]
        template_node = _PromptNode("Failed template: {{ prompt }}")

        jailbreak_count = fuzzer._process_scoring_results(
            context=basic_fuzzer_context,
            scores=low_scores,
            responses=responses,
            template_node=template_node,
            current_seed=parent_seed,
        )

        # Should not add template
        assert jailbreak_count == 0
        assert len(basic_fuzzer_context.new_prompt_nodes) == 0

        # Test with jailbreaks
        high_scores = [create_score(0.9) for _ in range(3)]
        successful_template_node = _PromptNode("Successful template: {{ prompt }}")

        jailbreak_count = fuzzer._process_scoring_results(
            context=basic_fuzzer_context,
            scores=high_scores,
            responses=responses,
            template_node=successful_template_node,
            current_seed=parent_seed,
        )

        # Should add template
        assert jailbreak_count == 3
        assert len(basic_fuzzer_context.new_prompt_nodes) == 1
        assert basic_fuzzer_context.new_prompt_nodes[0] == successful_template_node
        assert successful_template_node.parent == parent_seed

    @pytest.mark.parametrize(
        "threshold,score_value,expected_jailbreak",
        [
            (0.8, 0.9, True),
            (0.8, 0.8, True),
            (0.8, 0.79, False),
            (0.5, 0.6, True),
            (0.5, 0.4, False),
            (0.9, 0.89, False),
            (0.9, 0.91, True),
            (1.0, 1.0, True),
            (1.0, 0.99, False),
        ],
    )
    def test_custom_threshold_values(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_objective_scorer,
        create_score,
        threshold,
        score_value,
        expected_jailbreak,
    ):
        """Test jailbreak detection with custom threshold values."""
        scoring_config = AttackScoringConfig(
            objective_scorer=mock_objective_scorer,
            successful_objective_threshold=threshold,
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=scoring_config,
        )

        score = create_score(score_value)
        is_jailbreak = fuzzer._is_jailbreak(score)
        assert is_jailbreak == expected_jailbreak


@pytest.mark.usefixtures("patch_central_database")
class TestAttackResults:
    """Tests result creation and properties."""

    def test_create_attack_result_success(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test result when jailbreak goal is met."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            target_jailbreak_goal_count=2,
            max_query_limit=100,
        )

        # Create context with successful state
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={"test": "label"},
            total_target_query_count=30,
            total_jailbreak_count=3,  # Exceeds target of 2
            jailbreak_conversation_ids=["conv-1", "conv-2", "conv-3"],
            executed_turns=5,
        )

        # Add successful templates
        context.new_prompt_nodes = [
            _PromptNode("Success template 1: {{ prompt }}"),
            _PromptNode("Success template 2: {{ prompt }}"),
        ]

        # Create result
        result = fuzzer._create_attack_result(context)

        # Verify success outcome
        assert isinstance(result, FuzzerAttackResult)
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.outcome_reason is not None
        assert "Found 3 jailbreaks" in result.outcome_reason
        assert "target: 2" in result.outcome_reason
        assert result.objective == "Test objective"
        assert result.executed_turns == 5
        assert result.conversation_id == "conv-3"  # Last successful conversation

    def test_create_attack_result_query_limit(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test result when query limit is reached."""
        max_limit = 50
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            target_jailbreak_goal_count=10,  # High target
            max_query_limit=max_limit,
        )

        # Create context where query limit was hit
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={"test": "label"},
            total_target_query_count=max_limit,
            total_jailbreak_count=2,  # Below target
            jailbreak_conversation_ids=["conv-1", "conv-2"],
            executed_turns=8,
        )

        # Create result
        result = fuzzer._create_attack_result(context)

        # Verify failure outcome with query limit reason
        assert result.outcome == AttackOutcome.FAILURE
        assert result.outcome_reason is not None
        assert f"Query limit ({max_limit}) reached" in result.outcome_reason
        assert result.executed_turns == 8

    def test_attack_result_properties(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test successful_templates, jailbreak_conversation_ids, total_queries, templates_explored properties."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Create context with various data
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={"test": "label"},
            total_target_query_count=45,
            total_jailbreak_count=3,
            jailbreak_conversation_ids=["conv-1", "conv-2", "conv-3"],
            executed_turns=7,
        )

        # Add successful templates
        successful_templates = [
            "Template A: {{ prompt }}",
            "Template B: {{ prompt }}",
            "Template C: {{ prompt }}",
        ]
        context.new_prompt_nodes = [_PromptNode(template) for template in successful_templates]

        # Create result
        result = fuzzer._create_attack_result(context)

        # Test properties
        assert result.successful_templates == successful_templates
        assert result.jailbreak_conversation_ids == ["conv-1", "conv-2", "conv-3"]
        assert result.total_queries == 45
        assert result.templates_explored == 3

        # Test property setters
        result.successful_templates = ["New template"]
        assert result.successful_templates == ["New template"]

        result.jailbreak_conversation_ids = ["new-conv-1"]
        assert result.jailbreak_conversation_ids == ["new-conv-1"]

        result.total_queries = 100
        assert result.total_queries == 100

        result.templates_explored = 5
        assert result.templates_explored == 5

    def test_attack_result_metadata_serialization(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Verify metadata can be serialized/deserialized."""
        import json

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Create context
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={"test": "label"},
            total_target_query_count=25,
            total_jailbreak_count=2,
            jailbreak_conversation_ids=["conv-1", "conv-2"],
            executed_turns=4,
        )

        # Add successful templates
        context.new_prompt_nodes = [
            _PromptNode("Template 1: {{ prompt }}"),
            _PromptNode("Template 2: {{ prompt }}"),
        ]

        # Create result
        result = fuzzer._create_attack_result(context)

        # Serialize metadata
        metadata_json = json.dumps(result.metadata)

        # Deserialize metadata
        deserialized_metadata = json.loads(metadata_json)

        # Verify data integrity
        assert deserialized_metadata["successful_templates"] == result.successful_templates
        assert deserialized_metadata["jailbreak_conversation_ids"] == result.jailbreak_conversation_ids
        assert deserialized_metadata["total_queries"] == result.total_queries
        assert deserialized_metadata["templates_explored"] == result.templates_explored

    def test_result_with_no_jailbreaks(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test result creation when no jailbreaks were found."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            target_jailbreak_goal_count=1,
        )

        # Create context with no jailbreaks
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={"test": "label"},
            total_target_query_count=30,
            total_jailbreak_count=0,
            jailbreak_conversation_ids=[],
            executed_turns=5,
        )

        # Create result
        result = fuzzer._create_attack_result(context)

        # Verify failure outcome
        assert result.outcome == AttackOutcome.FAILURE
        assert result.outcome_reason is not None
        assert "Only found 0 jailbreaks" in result.outcome_reason
        assert result.conversation_id == ""  # No successful conversations
        assert result.successful_templates == []
        assert result.jailbreak_conversation_ids == []

    def test_result_with_last_response(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test getting last response from successful jailbreak."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Create mock response in memory
        conversation_id = "test-conv-123"

        # Create a mock response piece
        response_piece = PromptRequestPiece(
            role="assistant",
            original_value="This is the jailbreak response",
            converted_value="This is the jailbreak response",
            conversation_id=conversation_id,
        )

        # Mock memory to return response
        with patch.object(fuzzer._memory, "get_prompt_request_pieces", return_value=[response_piece]):
            # Create context
            context = FuzzerAttackContext(
                objective="Test objective",
                memory_labels={"test": "label"},
                total_jailbreak_count=1,
                jailbreak_conversation_ids=[conversation_id],
                executed_turns=1,
            )

            # Get last response
            last_response = fuzzer._get_last_response(context)

            assert last_response is not None
            assert last_response.role == "assistant"
            assert last_response.original_value == "This is the jailbreak response"

    def test_result_identifier_and_basic_fields(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test that basic result fields are populated correctly."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Get attack identifier
        attack_id = fuzzer.get_identifier()

        # Create context
        context = FuzzerAttackContext(
            objective="Specific test objective",
            memory_labels={"test": "label"},
            total_jailbreak_count=1,
            jailbreak_conversation_ids=["conv-xyz"],
            executed_turns=3,
        )

        # Create result
        result = fuzzer._create_attack_result(context)

        # Verify basic fields
        assert result.attack_identifier == attack_id
        assert result.objective == "Specific test objective"
        assert result.executed_turns == 3
        assert result.last_score is None  # Fuzzer doesn't track individual scores

    @pytest.mark.parametrize(
        "jailbreak_count,target_count,queries,query_limit,expected_outcome,expected_reason_contains",
        [
            (5, 3, 50, 100, AttackOutcome.SUCCESS, "Found 5 jailbreaks"),
            (2, 5, 50, 100, AttackOutcome.FAILURE, "Only found 2 jailbreaks"),
            (0, 1, 100, 100, AttackOutcome.FAILURE, "Query limit (100) reached"),
            (3, 3, 30, 100, AttackOutcome.SUCCESS, "Found 3 jailbreaks"),
            (1, 2, 100, 100, AttackOutcome.FAILURE, "Query limit (100) reached"),
        ],
    )
    def test_outcome_determination_scenarios(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        jailbreak_count,
        target_count,
        queries,
        query_limit,
        expected_outcome,
        expected_reason_contains,
    ):
        """Test various scenarios for outcome determination."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            target_jailbreak_goal_count=target_count,
            max_query_limit=query_limit,
        )

        # Create context
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={"test": "label"},
            total_target_query_count=queries,
            total_jailbreak_count=jailbreak_count,
            jailbreak_conversation_ids=[f"conv-{i}" for i in range(jailbreak_count)],
            executed_turns=queries // len(sample_prompts),
        )

        # Create result
        result = fuzzer._create_attack_result(context)

        # Verify outcome
        assert result.outcome == expected_outcome
        assert expected_reason_contains in result.outcome_reason

    def test_empty_successful_templates_handling(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test handling when no successful templates were found."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Create context with no successful templates
        context = FuzzerAttackContext(
            objective="Test objective",
            memory_labels={"test": "label"},
            total_target_query_count=20,
            total_jailbreak_count=0,
            jailbreak_conversation_ids=[],
            executed_turns=3,
            new_prompt_nodes=[],  # No successful templates
        )

        # Create result
        result = fuzzer._create_attack_result(context)

        # Verify empty templates
        assert result.successful_templates == []
        assert result.templates_explored == 0
        assert result.metadata["successful_templates"] == []
        assert result.metadata["templates_explored"] == 0


@pytest.mark.usefixtures("patch_central_database")
class TestEdgeCasesAndErrorHandling:
    """Tests edge cases and error scenarios."""

    def test_empty_prompts_list(
        self,
        mock_objective_target,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test behavior with no prompts."""
        with pytest.raises(ValueError, match="The initial prompts cannot be empty"):
            FuzzerAttack(
                objective_target=mock_objective_target,
                prompts=[],  # Empty prompts
                prompt_templates=sample_templates,
                template_converters=sample_converters,
                attack_scoring_config=attack_scoring_config,
            )

    def test_empty_templates_list(
        self,
        mock_objective_target,
        sample_prompts,
        sample_converters,
        attack_scoring_config,
    ):
        """Test behavior with no templates."""
        with pytest.raises(ValueError, match="The initial set of prompt templates cannot be empty"):
            FuzzerAttack(
                objective_target=mock_objective_target,
                prompts=sample_prompts,
                prompt_templates=[],  # Empty templates
                template_converters=sample_converters,
                attack_scoring_config=attack_scoring_config,
            )

    def test_single_prompt_single_template(
        self,
        mock_objective_target,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_prompt_response,
        create_score,
        create_converter_result,
    ):
        """Test minimal configuration with single prompt and template."""
        # Single items
        single_prompt = ["Single prompt"]
        single_template = ["Single template: {{ prompt }}"]

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=single_prompt,
            prompt_templates=single_template,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            target_jailbreak_goal_count=1,
        )

        # Mock converter
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(return_value=create_converter_result("Converted: {{ prompt }}"))

        # Mock responses and scores
        response = create_prompt_response("Response")
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(return_value=[response])

        score = create_score(0.9)  # Jailbreak
        attack_scoring_config.objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(return_value=[score])

        # Execute attack
        async def run_attack():
            await fuzzer._setup_async(context=basic_fuzzer_context)
            result = await fuzzer._perform_attack_async(context=basic_fuzzer_context)
            return result

        import asyncio

        result = asyncio.run(run_attack())

        # Verify successful execution
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.total_queries == 1
        assert len(result.jailbreak_conversation_ids) == 1

    def test_template_without_placeholder(
        self,
        mock_objective_target,
        sample_prompts,
        sample_converters,
        attack_scoring_config,
    ):
        """Test templates missing {{ prompt }} placeholder."""
        bad_templates = [
            "Template without placeholder",
            "Another bad template",
        ]

        with pytest.raises(MissingPromptPlaceholderException, match="Template missing placeholder"):
            FuzzerAttack(
                objective_target=mock_objective_target,
                prompts=sample_prompts,
                prompt_templates=bad_templates,
                template_converters=sample_converters,
                attack_scoring_config=attack_scoring_config,
            )

    @pytest.mark.asyncio
    async def test_concurrent_execution_safety(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        create_prompt_response,
        create_score,
        create_converter_result,
    ):
        """Test thread safety with multiple contexts."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            max_query_limit=len(sample_prompts),  # One iteration each
        )

        # Mock dependencies
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(return_value=create_converter_result("Converted: {{ prompt }}"))

        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(return_value=responses)

        scores = [create_score(0.5) for _ in range(len(sample_prompts))]
        attack_scoring_config.objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(return_value=scores)

        # Create multiple contexts
        contexts = []
        for i in range(3):
            context = FuzzerAttackContext(
                objective=f"Objective {i}",
                memory_labels={"context": str(i)},
            )
            contexts.append(context)

        # Execute concurrently
        async def execute_with_context(ctx):
            await fuzzer._setup_async(context=ctx)
            return await fuzzer._perform_attack_async(context=ctx)

        import asyncio

        results = await asyncio.gather(*[execute_with_context(ctx) for ctx in contexts])

        # Verify each execution was independent
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.objective == f"Objective {i}"
            assert result.outcome == AttackOutcome.FAILURE  # No jailbreaks found

    @pytest.mark.asyncio
    async def test_target_error_handling(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        basic_fuzzer_context,
        create_converter_result,
    ):
        """Test target failures and retries."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Mock converter
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(return_value=create_converter_result("Converted: {{ prompt }}"))

        # Mock normalizer to raise error
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
            side_effect=RuntimeError("Target connection failed")
        )

        # Setup
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Should propagate the error
        with pytest.raises(RuntimeError, match="Target connection failed"):
            await fuzzer._execute_attack_iteration_async(basic_fuzzer_context)

    @pytest.mark.asyncio
    async def test_scorer_error_handling(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        mock_objective_scorer,
        basic_fuzzer_context,
        create_prompt_response,
        create_converter_result,
    ):
        """Test scoring failures."""
        scoring_config = AttackScoringConfig(
            objective_scorer=mock_objective_scorer,
            successful_objective_threshold=0.8,
        )

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=scoring_config,
        )

        # Mock converter
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(return_value=create_converter_result("Converted: {{ prompt }}"))

        # Mock responses
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(return_value=responses)

        # Mock scorer to fail
        mock_objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(side_effect=Exception("Scorer failed"))

        # Setup
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Should propagate the error
        with pytest.raises(Exception, match="Scorer failed"):
            await fuzzer._execute_attack_iteration_async(basic_fuzzer_context)

    def test_invalid_query_limit(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test query limit validation."""
        # Query limit less than number of prompts
        with pytest.raises(ValueError, match="The query limit must be at least the number of prompts"):
            FuzzerAttack(
                objective_target=mock_objective_target,
                prompts=sample_prompts,
                prompt_templates=sample_templates,
                template_converters=sample_converters,
                attack_scoring_config=attack_scoring_config,
                max_query_limit=len(sample_prompts) - 1,
            )

    @pytest.mark.asyncio
    async def test_converter_retry_exhaustion(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        attack_scoring_config,
        basic_fuzzer_context,
        mock_scoring_target,
        create_converter_result,
    ):
        """Test when converter consistently fails to produce valid templates."""
        # Create converter that always removes placeholder
        bad_converter = FuzzerExpandConverter(converter_target=mock_scoring_target)
        bad_converter.update = MagicMock()
        bad_converter.convert_async = AsyncMock(return_value=create_converter_result("No placeholder in output"))

        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=[bad_converter],
            attack_scoring_config=attack_scoring_config,
        )

        # Setup
        await fuzzer._setup_async(context=basic_fuzzer_context)

        # Select a node
        current_seed, _ = fuzzer._select_template_with_mcts(basic_fuzzer_context)

        # Should raise after retries
        with pytest.raises(MissingPromptPlaceholderException, match="Converted template missing placeholder"):
            await fuzzer._apply_template_converter_async(context=basic_fuzzer_context, current_seed=current_seed)

    def test_extreme_parameter_values(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test with extreme parameter values."""
        # Test with very high values
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            frequency_weight=1000.0,  # Very high
            reward_penalty=0.99,  # Very high penalty
            minimum_reward=0.01,  # Very low minimum
            non_leaf_node_probability=0.99,  # Almost always stop at non-leaf
            batch_size=10000,  # Very large batch
            target_jailbreak_goal_count=1000000,  # Unrealistic target
            max_query_limit=1000000,  # Very high limit
        )

        # Should initialize without error
        assert fuzzer._mcts_explorer.frequency_weight == 1000.0
        assert fuzzer._mcts_explorer.reward_penalty == 0.99
        assert fuzzer._mcts_explorer.minimum_reward == 0.01
        assert fuzzer._mcts_explorer.non_leaf_node_probability == 0.99
        assert fuzzer._batch_size == 10000
        assert fuzzer._target_jailbreak_goal_count == 1000000
        assert fuzzer._max_query_limit == 1000000

    @pytest.mark.parametrize(
        "score_type,score_value",
        [
            ("dict", {"value": 0.5}),
            ("list", [0.5, 0.6, 0.7]),
            ("string", "0.5"),
            ("complex", complex(0.5, 0)),
            ("custom_object", object()),
        ],
    )
    def test_unusual_score_types(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
        score_type,
        score_value,
    ):
        """Test score normalization with unusual types."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Should handle gracefully and default to 0.0
        normalized = fuzzer._normalize_score_to_float(score_value)
        assert normalized == 0.0

    @pytest.mark.asyncio
    async def test_attack_with_no_successful_jailbreaks(
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
        """Test full attack execution when no jailbreaks are found."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
            target_jailbreak_goal_count=1,
            max_query_limit=len(sample_prompts) * 2,  # Allow 2 iterations
        )

        # Mock converter
        for converter in sample_converters:
            converter.update = MagicMock()
            converter.convert_async = AsyncMock(return_value=create_converter_result("Converted: {{ prompt }}"))

        # Mock responses
        responses = [create_prompt_response(f"Response {i}") for i in range(len(sample_prompts))]
        fuzzer._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(return_value=responses)

        # Mock low scores (no jailbreaks)
        low_scores = [create_score(0.3) for _ in range(len(sample_prompts))]
        attack_scoring_config.objective_scorer.score_prompts_with_tasks_batch_async = AsyncMock(return_value=low_scores)

        # Setup and execute
        await fuzzer._setup_async(context=basic_fuzzer_context)
        result = await fuzzer._perform_attack_async(context=basic_fuzzer_context)

        # Verify failure
        assert result.outcome == AttackOutcome.FAILURE
        assert result.total_queries == len(sample_prompts) * 2
        assert result.jailbreak_conversation_ids == []
        assert result.successful_templates == []
        assert result.outcome_reason is not None
        assert "Query limit" in result.outcome_reason

    def test_mcts_with_zero_step(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test MCTS algorithm with step=0 edge case."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Create nodes
        nodes = [_PromptNode(template) for template in sample_templates]

        # Select with step=0 (should handle log(0) case)
        selected_node, path = fuzzer._mcts_explorer.select_node(initial_nodes=nodes, step=0)

        # Should still select a node without error
        assert selected_node in nodes
        assert len(path) >= 1

    def test_generate_prompts_missing_parameter(
        self,
        mock_objective_target,
        sample_prompts,
        sample_templates,
        sample_converters,
        attack_scoring_config,
    ):
        """Test generate_prompts_from_template with missing parameter."""
        fuzzer = FuzzerAttack(
            objective_target=mock_objective_target,
            prompts=sample_prompts,
            prompt_templates=sample_templates,
            template_converters=sample_converters,
            attack_scoring_config=attack_scoring_config,
        )

        # Create template without 'prompt' parameter
        bad_template = SeedPrompt(value="Template without parameter", data_type="text", parameters=[])  # No parameters

        # Should raise ValueError
        with pytest.raises(ValueError, match="Template must have 'prompt' parameter"):
            fuzzer._generate_prompts_from_template(template=bad_template, prompts=sample_prompts)
