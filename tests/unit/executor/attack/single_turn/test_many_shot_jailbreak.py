# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import (
    AttackConverterConfig,
    AttackScoringConfig,
    ManyShotJailbreakAttack,
    SingleTurnAttackContext,
)
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    PromptRequestPiece,
    PromptRequestResponse,
    SeedPrompt,
)
from pyrit.prompt_converter import Base64Converter
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer


@pytest.fixture
def mock_objective_target():
    """Create a mock PromptTarget for testing"""
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = {"id": "mock_target_id"}
    return target


@pytest.fixture
def mock_template():
    """Create a mock SeedPrompt template"""
    template = MagicMock(spec=SeedPrompt)
    template.render_template_value.return_value = "Template with {prompt} and {examples}"
    return template


@pytest.fixture
def sample_many_shot_examples():
    """Sample many shot jailbreaking examples"""
    return [
        {"user": "question1", "assistant": "answer1"},
        {"user": "question2", "assistant": "answer2"},
        {"user": "question3", "assistant": "answer3"},
    ]


@pytest.fixture
def mock_scorer():
    """Create a mock true/false scorer"""
    scorer = MagicMock(spec=Scorer)
    scorer.scorer_type = "true_false"
    scorer.score_text_async = AsyncMock()
    return scorer


@pytest.fixture
def basic_context():
    """Create a basic context for testing"""
    return SingleTurnAttackContext(
        objective="How can I break into a computer system?", conversation_id=str(uuid.uuid4())
    )


@pytest.mark.usefixtures("patch_central_database")
class TestManyShotJailbreakAttackInitialization:
    """Tests for ManyShotJailbreakAttack initialization"""

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.fetch_many_shot_jailbreaking_dataset")
    def test_init_with_default_parameters(
        self, mock_fetch_dataset, mock_from_yaml, mock_objective_target, mock_template, sample_many_shot_examples
    ):
        """Test initialization with default parameters"""
        mock_from_yaml.return_value = mock_template
        mock_fetch_dataset.return_value = sample_many_shot_examples

        attack = ManyShotJailbreakAttack(objective_target=mock_objective_target)

        assert attack._objective_target == mock_objective_target
        assert attack._template == mock_template
        assert attack._examples == sample_many_shot_examples[:100]  # Default example_count is 100
        assert attack._max_attempts_on_failure == 0

        # Verify template was loaded from correct path
        mock_from_yaml.assert_called_once()
        args = mock_from_yaml.call_args[0]
        assert "many_shot_template.yaml" in str(args[0])

        # Verify dataset was fetched
        mock_fetch_dataset.assert_called_once()

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.fetch_many_shot_jailbreaking_dataset")
    def test_init_with_custom_example_count(
        self, mock_fetch_dataset, mock_from_yaml, mock_objective_target, mock_template, sample_many_shot_examples
    ):
        """Test initialization with custom example count"""
        mock_from_yaml.return_value = mock_template
        mock_fetch_dataset.return_value = sample_many_shot_examples

        attack = ManyShotJailbreakAttack(objective_target=mock_objective_target, example_count=2)

        assert len(attack._examples) == 2
        assert attack._examples == sample_many_shot_examples[:2]

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    def test_init_with_custom_examples(self, mock_from_yaml, mock_objective_target, mock_template):
        """Test initialization with custom many shot examples"""
        mock_from_yaml.return_value = mock_template
        custom_examples = [
            {"user": "Custom question 1", "assistant": "Custom answer 1"},
            {"user": "Custom question 2", "assistant": "Custom answer 2"},
        ]

        attack = ManyShotJailbreakAttack(
            objective_target=mock_objective_target, example_count=5, many_shot_examples=custom_examples
        )

        assert attack._examples == custom_examples

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    def test_init_with_all_parameters(self, mock_from_yaml, mock_objective_target, mock_template, mock_scorer):
        """Test initialization with all optional parameters"""
        mock_from_yaml.return_value = mock_template
        converter_config = AttackConverterConfig()
        scoring_config = AttackScoringConfig(objective_scorer=mock_scorer)
        prompt_normalizer = PromptNormalizer()
        custom_examples = [{"user": "test", "assistant": "response"}]

        attack = ManyShotJailbreakAttack(
            objective_target=mock_objective_target,
            attack_converter_config=converter_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=prompt_normalizer,
            max_attempts_on_failure=3,
            example_count=50,
            many_shot_examples=custom_examples,
        )

        assert attack._objective_target == mock_objective_target
        assert attack._objective_scorer == mock_scorer
        assert attack._prompt_normalizer == prompt_normalizer
        assert attack._max_attempts_on_failure == 3
        assert attack._examples == custom_examples

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.fetch_many_shot_jailbreaking_dataset")
    def test_init_raises_error_with_empty_examples(
        self, mock_fetch_dataset, mock_from_yaml, mock_objective_target, mock_template
    ):
        """Test that initialization raises error when no examples are available"""
        mock_from_yaml.return_value = mock_template
        mock_fetch_dataset.return_value = []

        with pytest.raises(ValueError, match="Many shot examples must be provided"):
            ManyShotJailbreakAttack(objective_target=mock_objective_target)

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    def test_init_raises_error_with_empty_custom_examples(self, mock_from_yaml, mock_objective_target, mock_template):
        """Test that initialization raises error with empty custom examples"""
        mock_from_yaml.return_value = mock_template

        with pytest.raises(ValueError, match="Many shot examples must be provided"):
            ManyShotJailbreakAttack(objective_target=mock_objective_target, many_shot_examples=[])


@pytest.mark.usefixtures("patch_central_database")
class TestManyShotJailbreakAttackValidation:
    """Tests for context validation"""

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.fetch_many_shot_jailbreaking_dataset")
    def test_validate_context_raises_error_with_prepended_conversation(
        self, mock_fetch_dataset, mock_from_yaml, mock_objective_target, mock_template, sample_many_shot_examples
    ):
        """Test that validation fails if prepended conversation is provided"""
        mock_from_yaml.return_value = mock_template
        mock_fetch_dataset.return_value = sample_many_shot_examples

        attack = ManyShotJailbreakAttack(objective_target=mock_objective_target)

        context = SingleTurnAttackContext(
            objective="Test objective", conversation_id=str(uuid.uuid4()), prepended_conversation=[MagicMock()]
        )

        with pytest.raises(ValueError, match="ManyShotJailbreakAttack does not support prepended conversations"):
            attack._validate_context(context=context)

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.fetch_many_shot_jailbreaking_dataset")
    def test_validate_context_passes_without_prepended_conversation(
        self,
        mock_fetch_dataset,
        mock_from_yaml,
        mock_objective_target,
        mock_template,
        sample_many_shot_examples,
        basic_context,
    ):
        """Test that validation passes when no prepended conversation is provided"""
        mock_from_yaml.return_value = mock_template
        mock_fetch_dataset.return_value = sample_many_shot_examples

        attack = ManyShotJailbreakAttack(objective_target=mock_objective_target)

        # Should not raise any exception
        attack._validate_context(context=basic_context)


@pytest.mark.usefixtures("patch_central_database")
class TestManyShotJailbreakAttackExecution:
    """Tests for attack execution"""

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.fetch_many_shot_jailbreaking_dataset")
    @pytest.mark.asyncio
    async def test_perform_attack_renders_template_correctly(
        self,
        mock_fetch_dataset,
        mock_from_yaml,
        mock_objective_target,
        mock_template,
        sample_many_shot_examples,
        basic_context,
    ):
        """Test that the attack correctly renders the template with examples and objective"""
        mock_from_yaml.return_value = mock_template
        mock_fetch_dataset.return_value = sample_many_shot_examples
        rendered_prompt = "Many shot jailbreak with examples and How can I break into a computer system?"
        mock_template.render_template_value.return_value = rendered_prompt

        attack = ManyShotJailbreakAttack(objective_target=mock_objective_target)

        # Mock the parent's perform_async
        with patch.object(
            ManyShotJailbreakAttack.__bases__[0], "_perform_async", new_callable=AsyncMock
        ) as mock_perform:
            mock_result = AttackResult(
                conversation_id=basic_context.conversation_id,
                objective=basic_context.objective,
                attack_identifier=attack.get_identifier(),
                outcome=AttackOutcome.SUCCESS,
            )
            mock_perform.return_value = mock_result

            result = await attack._perform_async(context=basic_context)

            # Verify template was rendered with correct parameters
            mock_template.render_template_value.assert_called_once_with(
                prompt=basic_context.objective, examples=sample_many_shot_examples[:100]
            )

            # Verify the seed prompt was set correctly
            assert basic_context.seed_prompt_group is not None
            assert len(basic_context.seed_prompt_group.prompts) == 1
            assert basic_context.seed_prompt_group.prompts[0].value == rendered_prompt
            assert basic_context.seed_prompt_group.prompts[0].data_type == "text"

            # Verify parent method was called
            mock_perform.assert_called_once_with(context=basic_context)
            assert result == mock_result

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.fetch_many_shot_jailbreaking_dataset")
    @pytest.mark.asyncio
    async def test_perform_attack_with_custom_examples(
        self, mock_fetch_dataset, mock_from_yaml, mock_objective_target, mock_template, basic_context
    ):
        """Test attack execution with custom examples"""
        mock_from_yaml.return_value = mock_template
        mock_fetch_dataset.return_value = []  # Won't be used since custom examples provided

        custom_examples = [
            {"user": "Custom harmful question", "assistant": "Custom response"},
            {"user": "Another question", "assistant": "Another response"},
        ]

        attack = ManyShotJailbreakAttack(objective_target=mock_objective_target, many_shot_examples=custom_examples)

        with patch.object(
            ManyShotJailbreakAttack.__bases__[0], "_perform_async", new_callable=AsyncMock
        ) as mock_perform:
            mock_result = AttackResult(
                conversation_id=basic_context.conversation_id,
                objective=basic_context.objective,
                attack_identifier=attack.get_identifier(),
                outcome=AttackOutcome.SUCCESS,
            )
            mock_perform.return_value = mock_result

            await attack._perform_async(context=basic_context)

            # Verify template was called with custom examples
            mock_template.render_template_value.assert_called_once_with(
                prompt=basic_context.objective, examples=custom_examples
            )


@pytest.mark.usefixtures("patch_central_database")
class TestManyShotJailbreakAttackLifecycle:
    """Tests for the complete attack lifecycle"""

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.fetch_many_shot_jailbreaking_dataset")
    @pytest.mark.asyncio
    async def test_execute_async_successful_lifecycle(
        self,
        mock_fetch_dataset,
        mock_from_yaml,
        mock_objective_target,
        mock_template,
        sample_many_shot_examples,
        basic_context,
    ):
        """Test the complete attack lifecycle"""
        mock_from_yaml.return_value = mock_template
        mock_fetch_dataset.return_value = sample_many_shot_examples

        attack = ManyShotJailbreakAttack(objective_target=mock_objective_target)

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

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.fetch_many_shot_jailbreaking_dataset")
    @pytest.mark.asyncio
    async def test_execute_async_validation_failure_prevents_execution(
        self,
        mock_fetch_dataset,
        mock_from_yaml,
        mock_objective_target,
        mock_template,
        sample_many_shot_examples,
        basic_context,
    ):
        """Test that validation failure prevents attack execution"""
        mock_from_yaml.return_value = mock_template
        mock_fetch_dataset.return_value = sample_many_shot_examples

        attack = ManyShotJailbreakAttack(objective_target=mock_objective_target)

        # Context with prepended conversation (not allowed)
        basic_context.prepended_conversation = [
            PromptRequestResponse(
                request_pieces=[PromptRequestPiece(role="user", original_value="Test prepended conversation")]
            )
        ]

        attack._setup_async = AsyncMock()
        attack._perform_async = AsyncMock()
        attack._teardown_async = AsyncMock()

        with pytest.raises(ValueError) as exc_info:
            await attack.execute_with_context_async(context=basic_context)

        # Verify error details
        assert "Strategy context validation failed for ManyShotJailbreakAttack" in str(exc_info.value)

        # Verify other methods weren't called
        attack._setup_async.assert_not_called()
        attack._perform_async.assert_not_called()
        attack._teardown_async.assert_not_called()


@pytest.mark.usefixtures("patch_central_database")
class TestManyShotJailbreakAttackWithConverters:
    """Tests for attack with converters"""

    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.SeedPrompt.from_yaml_file")
    @patch("pyrit.executor.attack.single_turn.many_shot_jailbreak.fetch_many_shot_jailbreaking_dataset")
    @pytest.mark.asyncio
    async def test_attack_with_request_converters(
        self,
        mock_fetch_dataset,
        mock_from_yaml,
        mock_objective_target,
        mock_template,
        sample_many_shot_examples,
        basic_context,
    ):
        """Test that the attack works with request converters"""
        mock_from_yaml.return_value = mock_template
        mock_fetch_dataset.return_value = sample_many_shot_examples

        converter_config = AttackConverterConfig(
            request_converters=PromptConverterConfiguration.from_converters(converters=[Base64Converter()])
        )

        attack = ManyShotJailbreakAttack(
            objective_target=mock_objective_target, attack_converter_config=converter_config
        )

        # Verify converter configuration was preserved
        assert len(attack._request_converters) == 1
        assert isinstance(attack._request_converters[0].converters[0], Base64Converter)

        # Mock the parent's perform_async to verify it gets called
        with patch.object(
            ManyShotJailbreakAttack.__bases__[0], "_perform_async", new_callable=AsyncMock
        ) as mock_perform:
            mock_result = AttackResult(
                conversation_id=basic_context.conversation_id,
                objective=basic_context.objective,
                attack_identifier=attack.get_identifier(),
                outcome=AttackOutcome.SUCCESS,
            )
            mock_perform.return_value = mock_result

            result = await attack._perform_async(context=basic_context)

            # Verify seed prompt group was created
            assert basic_context.seed_prompt_group is not None
            assert len(basic_context.seed_prompt_group.prompts) == 1

            # Verify parent method was called
            mock_perform.assert_called_once_with(context=basic_context)
            assert result == mock_result
