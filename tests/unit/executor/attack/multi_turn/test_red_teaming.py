# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from pathlib import Path
from typing import Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
    ConversationSession,
    ConversationState,
    MultiTurnAttackContext,
    RedTeamingAttack,
    RTASystemPromptPaths,
)
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    ConversationReference,
    ConversationType,
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    SeedPrompt,
)
from pyrit.models.score import ScoreType
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import Scorer


@pytest.fixture
def mock_objective_target() -> MagicMock:
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = {"__type__": "MockTarget", "__module__": "test_module"}
    return target


@pytest.fixture
def mock_adversarial_chat() -> MagicMock:
    chat = MagicMock(spec=PromptChatTarget)
    chat.send_prompt_async = AsyncMock()
    chat.set_system_prompt = MagicMock()
    chat.get_identifier.return_value = {"__type__": "MockChatTarget", "__module__": "test_module"}
    return chat


@pytest.fixture
def mock_objective_scorer() -> MagicMock:
    scorer = MagicMock(spec=Scorer)
    scorer.scorer_type = "true_false"
    scorer.score_async = AsyncMock()
    scorer.get_identifier.return_value = {"__type__": "MockScorer", "__module__": "test_module"}
    return scorer


@pytest.fixture
def mock_prompt_normalizer() -> MagicMock:
    normalizer = MagicMock(spec=PromptNormalizer)
    normalizer.send_prompt_async = AsyncMock()
    return normalizer


@pytest.fixture
def basic_context() -> MultiTurnAttackContext:
    return MultiTurnAttackContext(
        objective="Test objective",
        session=ConversationSession(),
    )


@pytest.fixture
def sample_response() -> PromptRequestResponse:
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value="Test response",
                original_value_data_type="text",
                converted_value="Test response",
                converted_value_data_type="text",
            )
        ]
    )


@pytest.fixture
def success_score() -> Score:
    return Score(
        score_type="true_false",
        score_value="true",
        score_category="test",
        score_value_description="Test success score",
        score_rationale="Test rationale for success",
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
        scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
    )


@pytest.fixture
def failure_score() -> Score:
    return Score(
        score_type="true_false",
        score_value="false",
        score_category="test",
        score_value_description="Test failure score",
        score_rationale="Test rationale for failure",
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
        scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
    )


@pytest.fixture
def float_score() -> Score:
    return Score(
        score_type="float_scale",
        score_value="0.9",
        score_category="test",
        score_value_description="High score",
        score_rationale="Test rationale for high score",
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
        scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
    )


@pytest.mark.usefixtures("patch_central_database")
class TestRedTeamingAttackInitialization:
    """Tests for RedTeamingAttack initialization and configuration"""

    def test_init_with_minimal_required_parameters(
        self, mock_objective_target: MagicMock, mock_objective_scorer: MagicMock, mock_adversarial_chat: MagicMock
    ):
        """Test that attack initializes correctly with only required parameters."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        assert attack._objective_target == mock_objective_target
        assert attack._objective_scorer == mock_objective_scorer
        assert attack._adversarial_chat == mock_adversarial_chat
        assert isinstance(attack._prompt_normalizer, PromptNormalizer)
        assert attack._max_turns == 10  # Default value

    @pytest.mark.parametrize(
        "system_prompt_path",
        [
            RTASystemPromptPaths.TEXT_GENERATION.value,
            RTASystemPromptPaths.IMAGE_GENERATION.value,
            RTASystemPromptPaths.NAIVE_CRESCENDO.value,
            RTASystemPromptPaths.VIOLENT_DURIAN.value,
            RTASystemPromptPaths.CRUCIBLE.value,
        ],
    )
    def test_init_with_different_system_prompts(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        system_prompt_path: Path,
    ):
        """Test that attack initializes correctly with different system prompt paths."""
        adversarial_config = AttackAdversarialConfig(
            target=mock_adversarial_chat, system_prompt_path=system_prompt_path
        )
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        assert attack._adversarial_chat_system_prompt_template is not None
        assert attack._adversarial_chat_system_prompt_template.parameters is not None
        assert "objective" in attack._adversarial_chat_system_prompt_template.parameters

    @pytest.mark.parametrize(
        "seed_prompt,expected_value,expected_type",
        [
            ("Custom seed prompt", "Custom seed prompt", str),
            (SeedPrompt(value="Custom seed", data_type="text"), "Custom seed", SeedPrompt),
        ],
    )
    def test_init_with_seed_prompt_variations(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        seed_prompt: Union[str, SeedPrompt],
        expected_value: str,
        expected_type: type,
    ):
        """Test that attack handles different seed prompt input types correctly."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat, seed_prompt=seed_prompt)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        assert attack._adversarial_chat_seed_prompt.value == expected_value
        if expected_type == str:
            assert attack._adversarial_chat_seed_prompt.data_type == "text"

    def test_init_with_invalid_system_prompt_path_raises_error(
        self, mock_objective_target: MagicMock, mock_objective_scorer: MagicMock, mock_adversarial_chat: MagicMock
    ):
        """Test that invalid system prompt path raises FileNotFoundError."""
        adversarial_config = AttackAdversarialConfig(
            target=mock_adversarial_chat, system_prompt_path="nonexistent_file.yaml"
        )
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        with pytest.raises(FileNotFoundError):
            RedTeamingAttack(
                objective_target=mock_objective_target,
                attack_adversarial_config=adversarial_config,
                attack_scoring_config=scoring_config,
            )

    def test_init_with_all_custom_configurations(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
    ):
        """Test that attack initializes correctly with all custom configurations."""
        converter_config = AttackConverterConfig()
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer, use_score_as_feedback=True)
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_converter_config=converter_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
            max_turns=20,  # Custom max turns
        )

        assert attack._request_converters == converter_config.request_converters
        assert attack._response_converters == converter_config.response_converters
        assert attack._prompt_normalizer == mock_prompt_normalizer
        assert attack._max_turns == 20

    def test_init_without_objective_scorer_raises_error(
        self, mock_objective_target: MagicMock, mock_adversarial_chat: MagicMock
    ):
        """Test that initialization without objective scorer raises ValueError."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig()  # No objective_scorer

        with pytest.raises(ValueError, match="Objective scorer must be provided"):
            RedTeamingAttack(
                objective_target=mock_objective_target,
                attack_adversarial_config=adversarial_config,
                attack_scoring_config=scoring_config,
            )


@pytest.mark.usefixtures("patch_central_database")
class TestContextCreation:
    """Tests for context creation from parameters"""

    @pytest.mark.asyncio
    async def test_execute_async_creates_context_properly(
        self, mock_objective_target: MagicMock, mock_objective_scorer: MagicMock, mock_adversarial_chat: MagicMock
    ):
        """Test that execute_async creates context properly."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            max_turns=15,
        )

        # Mock the execution methods
        with patch.object(attack, "_validate_context") as mock_validate:
            with patch.object(attack, "_setup_async", new_callable=AsyncMock):
                with patch.object(attack, "_perform_async", new_callable=AsyncMock) as mock_perform:
                    with patch.object(attack, "_teardown_async", new_callable=AsyncMock):
                        captured_context = None

                        async def capture_context(*args, **kwargs):
                            # Capture the context that was created
                            nonlocal captured_context
                            captured_context = kwargs.get("context")
                            return AttackResult(
                                conversation_id="test-id",
                                objective="Test objective",
                                attack_identifier=attack.get_identifier(),
                                outcome=AttackOutcome.SUCCESS,
                                executed_turns=1,
                            )

                        mock_perform.side_effect = capture_context

                        # Execute
                        await attack.execute_async(
                            objective="Test objective",
                            memory_labels={"test": "label"},
                        )

                        # Verify the captured context
                        assert captured_context is not None
                        assert captured_context.objective == "Test objective"
                        assert (
                            captured_context.prepended_conversation is None
                            or captured_context.prepended_conversation == []
                        )
                        assert captured_context.memory_labels == {"test": "label"}

                        # Verify that validation was called
                        mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_async_with_custom_prompt(
        self, mock_objective_target: MagicMock, mock_objective_scorer: MagicMock, mock_adversarial_chat: MagicMock
    ):
        """Test context creation with custom prompt."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        with patch.object(attack, "_validate_context"):
            with patch.object(attack, "_setup_async", new_callable=AsyncMock):
                with patch.object(attack, "_perform_async", new_callable=AsyncMock) as mock_perform:
                    with patch.object(attack, "_teardown_async", new_callable=AsyncMock):
                        captured_context = None

                        async def capture_context(*args, **kwargs):
                            nonlocal captured_context
                            captured_context = kwargs.get("context")
                            return AttackResult(
                                conversation_id="test-id",
                                objective="Test objective",
                                attack_identifier=attack.get_identifier(),
                                outcome=AttackOutcome.SUCCESS,
                                executed_turns=1,
                            )

                        mock_perform.side_effect = capture_context

                        # Execute with custom prompt
                        await attack.execute_async(
                            objective="Test objective",
                            custom_prompt="My custom prompt",
                        )

                        # Verify the captured context
                        assert captured_context is not None
                        assert captured_context.custom_prompt == "My custom prompt"

    @pytest.mark.asyncio
    async def test_execute_async_invalid_custom_prompt_type(
        self, mock_objective_target: MagicMock, mock_objective_scorer: MagicMock, mock_adversarial_chat: MagicMock
    ):
        """Test that non-string custom prompt raises ValueError."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        # Should raise TypeError during parameter validation
        with pytest.raises(TypeError, match="Parameter 'custom_prompt' must be of type str"):
            await attack.execute_async(
                objective="Test objective",
                custom_prompt=123,  # Invalid type
            )


@pytest.mark.usefixtures("patch_central_database")
class TestContextValidation:
    """Tests for context validation logic"""

    @pytest.mark.parametrize(
        "objective,max_turns,executed_turns,expected_error",
        [
            ("", 5, 0, "Attack objective must be provided"),
            ("Test objective", 5, 5, "Already exceeded max turns"),
            ("Test objective", 5, 6, "Already exceeded max turns"),
        ],
    )
    def test_validate_context_raises_errors(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        objective: str,
        max_turns: int,
        executed_turns: int,
        expected_error: str,
    ):
        """Test that context validation raises appropriate errors for invalid inputs."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            max_turns=max_turns,
        )
        context = MultiTurnAttackContext(objective=objective, executed_turns=executed_turns)

        with pytest.raises(ValueError, match=expected_error):
            attack._validate_context(context=context)

    def test_validate_context_with_valid_context(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that valid context passes validation without errors."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            max_turns=10,
        )
        attack._validate_context(context=basic_context)  # Should not raise

    def test_init_with_invalid_max_turns(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test that initialization with invalid max_turns raises ValueError."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        with pytest.raises(ValueError, match="Maximum turns must be a positive integer"):
            RedTeamingAttack(
                objective_target=mock_objective_target,
                attack_adversarial_config=adversarial_config,
                attack_scoring_config=scoring_config,
                max_turns=0,
            )


@pytest.mark.usefixtures("patch_central_database")
class TestSetupPhase:
    """Tests for the setup phase of the attack."""

    @pytest.mark.asyncio
    async def test_setup_initializes_conversation_session(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that setup correctly initializes a conversation session."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        # Mock conversation manager
        mock_state = ConversationState(turn_count=0)
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=basic_context)

        assert basic_context.session is not None
        assert isinstance(basic_context.session, ConversationSession)

    @pytest.mark.asyncio
    async def test_setup_updates_turn_count_from_prepended_conversation(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that setup updates turn count based on prepended conversation state."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        # Mock conversation state with existing turns
        mock_state = ConversationState(turn_count=3)
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=basic_context)

        assert basic_context.executed_turns == 3

    @pytest.mark.asyncio
    async def test_setup_merges_memory_labels_correctly(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that memory labels from attack and context are merged correctly."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        # Add memory labels to both attack and context
        attack._memory_labels = {"strategy_label": "strategy_value", "common": "strategy"}
        basic_context.memory_labels = {"context_label": "context_value", "common": "context"}

        # Mock conversation manager
        mock_state = ConversationState(turn_count=0)
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=basic_context)

        # Context labels should override strategy labels for common keys
        assert basic_context.memory_labels == {
            "strategy_label": "strategy_value",
            "context_label": "context_value",
            "common": "context",
        }

    @pytest.mark.asyncio
    async def test_setup_sets_adversarial_chat_system_prompt(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that setup correctly sets the adversarial chat system prompt."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        # Mock conversation manager
        mock_state = ConversationState(turn_count=0)
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=basic_context)

        # Verify system prompt was set
        mock_adversarial_chat.set_system_prompt.assert_called_once()
        call_args = mock_adversarial_chat.set_system_prompt.call_args
        assert "Test objective" in call_args.kwargs["system_prompt"]
        assert call_args.kwargs["conversation_id"] == basic_context.session.adversarial_chat_conversation_id

    @pytest.mark.asyncio
    async def test_setup_retrieves_last_score_matching_scorer_type(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        success_score: Score,
    ):
        """Test that setup correctly retrieves the last score matching the objective scorer type."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        # Mock conversation state with scores
        other_score = Score(
            score_type="float_scale",
            score_value="0.5",
            score_category="other",
            score_value_description="Other score",
            score_rationale="Other rationale",
            score_metadata="{}",
            prompt_request_response_id=str(uuid.uuid4()),
            scorer_class_identifier={"__type__": "OtherScorer", "__module__": "test_module"},
        )

        mock_state = ConversationState(
            turn_count=1,
            last_assistant_message_scores=[other_score, success_score],
        )
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=basic_context)

        assert basic_context.last_score == success_score


@pytest.mark.usefixtures("patch_central_database")
class TestPromptGeneration:
    """Tests for prompt generation logic"""

    @pytest.mark.asyncio
    async def test_generate_next_prompt_uses_custom_prompt_on_first_turn(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that custom prompt is used when provided and cleared after use."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        first_prompt = "Custom first prompt"
        basic_context.executed_turns = 0
        basic_context.custom_prompt = first_prompt

        result = await attack._generate_next_prompt_async(context=basic_context)

        assert result == first_prompt
        assert basic_context.custom_prompt is None  # Should be cleared after use
        # Should not call adversarial chat
        mock_prompt_normalizer.send_prompt_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_next_prompt_uses_custom_prompt_regardless_of_turn(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that custom prompt is used even when executed_turns > 0."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        custom_prompt = "Custom prompt at turn 1"
        basic_context.executed_turns = 1  # Not first turn
        basic_context.custom_prompt = custom_prompt

        result = await attack._generate_next_prompt_async(context=basic_context)

        assert result == custom_prompt
        assert basic_context.custom_prompt is None  # Should be cleared after use
        # Should not call adversarial chat
        mock_prompt_normalizer.send_prompt_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_next_prompt_uses_adversarial_chat_after_first_turn(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
    ):
        """Test that adversarial chat is used to generate prompts when no custom prompt."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.executed_turns = 1
        basic_context.custom_prompt = None  # No custom prompt
        mock_prompt_normalizer.send_prompt_async.return_value = sample_response

        # Mock build_adversarial_prompt
        with patch.object(attack, "_build_adversarial_prompt", new_callable=AsyncMock, return_value="Built prompt"):
            result = await attack._generate_next_prompt_async(context=basic_context)

        assert result == sample_response.get_value()
        mock_prompt_normalizer.send_prompt_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_next_prompt_raises_on_none_response(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that ValueError is raised when adversarial chat returns None."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.executed_turns = 1
        mock_prompt_normalizer.send_prompt_async.return_value = None

        # Mock build_adversarial_prompt
        with patch.object(attack, "_build_adversarial_prompt", new_callable=AsyncMock, return_value="Built prompt"):
            with pytest.raises(ValueError, match="Received no response from adversarial chat"):
                await attack._generate_next_prompt_async(context=basic_context)


@pytest.mark.usefixtures("patch_central_database")
class TestAdversarialPromptBuilding:
    """Tests for building adversarial prompts."""

    @pytest.mark.asyncio
    async def test_build_adversarial_prompt_returns_seed_when_no_response(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that the seed prompt is returned when no previous response exists."""
        seed = "Initial seed"
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat, seed_prompt=seed)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.last_response = None
        result = await attack._build_adversarial_prompt(basic_context)

        assert result == seed

    @pytest.mark.parametrize(
        "data_type,converted_value,has_error,is_blocked,expected_result",
        [
            ("text", "Normal response", False, False, "Normal response"),
            ("text", "Response with feedback", False, False, "Response with feedback"),
            ("error", "Error message", True, False, "Request to target failed: Error message"),
            (
                "text",
                "",
                True,
                True,
                "Request to target failed: blocked. Please rewrite your prompt to avoid getting blocked next time.",
            ),
            ("text", "", False, False, "The previous response was empty. Please continue."),
        ],
    )
    async def test_handle_adversarial_text_response(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        data_type: str,
        converted_value: str,
        has_error: bool,
        is_blocked: bool,
        expected_result: str,
    ):
        """Test handling of various text response scenarios."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        response_piece = MagicMock(spec=PromptRequestPiece)
        response_piece.converted_value_data_type = data_type
        response_piece.converted_value = converted_value
        response_piece.has_error.return_value = has_error
        response_piece.is_blocked.return_value = is_blocked
        response_piece.response_error = "Error message"

        basic_context.last_response = MagicMock(spec=PromptRequestResponse)
        basic_context.last_response.get_piece.return_value = response_piece
        basic_context.last_score = None

        result = attack._handle_adversarial_text_response(context=basic_context)

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_handle_adversarial_text_response_with_feedback(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        success_score: Score,
    ):
        """Test that scoring feedback is appended to text responses when enabled."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer, use_score_as_feedback=True)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        response_piece = MagicMock(spec=PromptRequestPiece)
        response_piece.converted_value_data_type = "text"
        response_piece.converted_value = "Target response"
        response_piece.has_error.return_value = False

        basic_context.last_response = MagicMock(spec=PromptRequestResponse)
        basic_context.last_response.get_piece.return_value = response_piece
        basic_context.last_score = success_score

        result = attack._handle_adversarial_text_response(context=basic_context)

        assert "Target response" in result
        assert success_score.score_rationale in result

    def test_handle_adversarial_text_response_no_response(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test handling when no last response is available."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.last_response = None

        result = attack._handle_adversarial_text_response(context=basic_context)

        assert result == "No response available. Please continue."

    def test_handle_adversarial_file_response_raises_on_error(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that file response with error raises RuntimeError."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        response_piece = MagicMock(spec=PromptRequestPiece)
        response_piece.converted_value_data_type = "image_path"
        response_piece.has_error.return_value = True
        response_piece.response_error = "File error"

        basic_context.last_response = MagicMock(spec=PromptRequestResponse)
        basic_context.last_response.get_piece.return_value = response_piece

        with pytest.raises(RuntimeError, match="Request to target failed.*File error"):
            attack._handle_adversarial_file_response(context=basic_context)

    def test_handle_adversarial_file_response_without_feedback_raises(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that file response without feedback enabled raises ValueError."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer, use_score_as_feedback=False)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        response_piece = MagicMock(spec=PromptRequestPiece)
        response_piece.converted_value_data_type = "image_path"
        response_piece.has_error.return_value = False

        basic_context.last_response = MagicMock(spec=PromptRequestResponse)
        basic_context.last_response.get_piece.return_value = response_piece

        with pytest.raises(ValueError, match="use_score_as_feedback flag is set to False"):
            attack._handle_adversarial_file_response(context=basic_context)

    def test_handle_adversarial_file_response_with_feedback(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        success_score: Score,
    ):
        """Test that file response with feedback returns score rationale."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer, use_score_as_feedback=True)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        response_piece = MagicMock(spec=PromptRequestPiece)
        response_piece.converted_value_data_type = "image_path"
        response_piece.has_error.return_value = False

        basic_context.last_response = MagicMock(spec=PromptRequestResponse)
        basic_context.last_response.get_piece.return_value = response_piece
        basic_context.last_score = success_score

        result = attack._handle_adversarial_file_response(context=basic_context)

        assert result == success_score.score_rationale

    def test_handle_adversarial_file_response_no_response(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test handling when no last response is available for file response."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer, use_score_as_feedback=True)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.last_response = None

        result = attack._handle_adversarial_file_response(context=basic_context)

        assert result == "No response available. Please continue."


@pytest.mark.usefixtures("patch_central_database")
class TestResponseScoring:
    """Tests for response scoring logic."""

    @pytest.mark.asyncio
    async def test_score_response_successful(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
        success_score: Score,
    ):
        """Test successful scoring of a response."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.last_response = sample_response
        basic_context.objective = "Test objective"

        # Mock the Scorer.score_response_async method
        with patch(
            "pyrit.score.Scorer.score_response_async",
            new_callable=AsyncMock,
            return_value={"objective_scores": [success_score], "auxiliary_scores": []},
        ):
            result = await attack._score_response_async(context=basic_context)

        assert result == success_score

    @pytest.mark.asyncio
    async def test_score_response_returns_none_for_blocked(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that blocked responses return None without scoring."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        response_piece = MagicMock(spec=PromptRequestPiece)
        response_piece.is_blocked.return_value = True

        basic_context.last_response = MagicMock(spec=PromptRequestResponse)
        basic_context.last_response.get_piece.return_value = response_piece

        result = await attack._score_response_async(context=basic_context)

        assert result is None

    @pytest.mark.asyncio
    async def test_score_response_returns_none_when_no_response(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that None is returned when no response is available."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.last_response = None

        result = await attack._score_response_async(context=basic_context)

        assert result is None


@pytest.mark.usefixtures("patch_central_database")
class TestAttackExecution:
    """Tests for the main attack execution logic."""

    @pytest.mark.parametrize(
        "scorer_type,score_value,threshold,expected_achieved",
        [
            ("true_false", "true", 0.8, True),
            ("true_false", "false", 0.8, False),
            ("float_scale", "0.9", 0.8, True),
            ("float_scale", "0.7", 0.8, False),
            ("float_scale", "0.8", 0.8, True),  # Edge case: equal to threshold
        ],
    )
    @pytest.mark.asyncio
    async def test_perform_attack_with_different_scoring_thresholds(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
        scorer_type: ScoreType,
        score_value: str,
        threshold: float,
        expected_achieved: bool,
    ):
        """Test attack execution with different scoring thresholds."""
        mock_objective_scorer.scorer_type = scorer_type

        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(
            objective_scorer=mock_objective_scorer, successful_objective_threshold=threshold
        )

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Create appropriate score
        score = Score(
            score_type=scorer_type,
            score_value=score_value,
            score_category="test",
            score_value_description=f"Score: {score_value}",
            score_rationale="Test rationale",
            score_metadata="{}",
            prompt_request_response_id=str(uuid.uuid4()),
            scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
        )

        # Mock methods
        with patch.object(attack, "_generate_next_prompt_async", new_callable=AsyncMock, return_value="Attack prompt"):
            with patch.object(
                attack,
                "_send_prompt_to_objective_target_async",
                new_callable=AsyncMock,
                return_value=sample_response,
            ):
                with patch.object(attack, "_score_response_async", new_callable=AsyncMock, return_value=score):
                    result = await attack._perform_async(context=basic_context)

        assert (result.outcome == AttackOutcome.SUCCESS) == expected_achieved

    @pytest.mark.asyncio
    async def test_perform_attack_reaches_max_turns(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
        failure_score: Score,
    ):
        """Test that attack stops after reaching max turns."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
            max_turns=3,  # Set max turns in attack init
        )

        # Mock methods to always fail
        with patch.object(
            attack, "_generate_next_prompt_async", new_callable=AsyncMock, return_value="Attack prompt"
        ) as mock_generate:
            with patch.object(
                attack,
                "_send_prompt_to_objective_target_async",
                new_callable=AsyncMock,
                return_value=sample_response,
            ) as mock_send:
                with patch.object(
                    attack, "_score_response_async", new_callable=AsyncMock, return_value=failure_score
                ) as mock_score:
                    result = await attack._perform_async(context=basic_context)

        assert result.outcome == AttackOutcome.FAILURE
        assert result.executed_turns == 3
        assert mock_generate.call_count == 3
        assert mock_send.call_count == 3
        assert mock_score.call_count == 3


@pytest.mark.usefixtures("patch_central_database")
class TestAttackLifecycle:
    """Tests for the complete attack lifecycle (execute_async)"""

    @pytest.mark.asyncio
    async def test_execute_async_successful_lifecycle(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        sample_response: PromptRequestResponse,
        success_score: Score,
    ):
        """Test successful execution of complete attack lifecycle using execute_async."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            max_turns=5,
        )

        # Mock all lifecycle methods
        with patch.object(attack, "_validate_context"):
            with patch.object(attack, "_setup_async", new_callable=AsyncMock):
                with patch.object(attack, "_perform_async", new_callable=AsyncMock) as mock_perform:
                    with patch.object(attack, "_teardown_async", new_callable=AsyncMock):
                        # Configure the return value for _perform_async
                        mock_perform.return_value = AttackResult(
                            conversation_id="test-conversation-id",
                            objective="Test objective",
                            attack_identifier=attack.get_identifier(),
                            outcome=AttackOutcome.SUCCESS,
                            executed_turns=1,
                            last_response=sample_response.get_piece(),
                            last_score=success_score,
                        )

                        # Execute using execute_async
                        result = await attack.execute_async(
                            objective="Test objective",
                        )

        # Verify result and proper execution order
        assert isinstance(result, AttackResult)
        assert result.outcome == AttackOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_async_validation_failure_prevents_execution(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test that validation failure prevents attack execution when using execute_async."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        # Mock validation to fail
        with patch.object(attack, "_validate_context", side_effect=ValueError("Invalid context")) as mock_validate:
            with patch.object(attack, "_setup_async", new_callable=AsyncMock) as mock_setup:
                with patch.object(attack, "_perform_async", new_callable=AsyncMock) as mock_perform:
                    with patch.object(attack, "_teardown_async", new_callable=AsyncMock) as mock_teardown:
                        # Should raise ValueError
                        with pytest.raises(ValueError) as exc_info:
                            await attack.execute_async(
                                objective="Test objective",
                            )

                        # Verify error details
                        assert "Strategy context validation failed for RedTeamingAttack" in str(
                            exc_info.value
                        )  # Verify only validation was attempted
        mock_validate.assert_called_once()
        mock_setup.assert_not_called()
        mock_perform.assert_not_called()
        mock_teardown.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_with_context_async_successful(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
        success_score: Score,
    ):
        """Test successful execution using execute_with_context_async."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        # Mock all lifecycle methods
        with patch.object(attack, "_validate_context"):
            with patch.object(attack, "_setup_async", new_callable=AsyncMock):
                with patch.object(attack, "_perform_async", new_callable=AsyncMock) as mock_perform:
                    with patch.object(attack, "_teardown_async", new_callable=AsyncMock):
                        # Configure the return value for _perform_async
                        mock_perform.return_value = AttackResult(
                            conversation_id=basic_context.session.conversation_id,
                            objective=basic_context.objective,
                            attack_identifier=attack.get_identifier(),
                            outcome=AttackOutcome.SUCCESS,
                            executed_turns=1,
                            last_response=sample_response.get_piece(),
                            last_score=success_score,
                        )

                        # Execute using execute_with_context_async
                        result = await attack.execute_with_context_async(context=basic_context)

        # Verify result
        assert isinstance(result, AttackResult)
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.objective == basic_context.objective

    @pytest.mark.asyncio
    async def test_teardown_async_is_noop(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that teardown completes without errors."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        # Should complete without error
        await attack._teardown_async(context=basic_context)
        # No assertions needed - we just want to ensure it runs without exceptions


@pytest.mark.usefixtures("patch_central_database")
class TestRedTeamingConversationTracking:
    """Test that adversarial chat conversation IDs are properly tracked."""

    @pytest.mark.asyncio
    async def test_setup_tracks_adversarial_chat_conversation_id(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that setup adds the adversarial chat conversation ID to the context's tracking."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            attack_scoring_config=AttackScoringConfig(objective_scorer=mock_objective_scorer),
        )

        # Mock the conversation manager to return a state
        with patch.object(attack._conversation_manager, "update_conversation_state_async") as mock_update:
            mock_update.return_value = ConversationState(
                turn_count=0, last_user_message="", last_assistant_message_scores=[]
            )

            # Run setup
            await attack._setup_async(context=basic_context)

            # Verify the adversarial chat conversation ID is tracked
            assert (
                ConversationReference(
                    conversation_id=basic_context.session.adversarial_chat_conversation_id,
                    conversation_type=ConversationType.ADVERSARIAL,
                )
                in basic_context.related_conversations
            )

    @pytest.mark.asyncio
    async def test_attack_result_includes_adversarial_chat_conversation_ids(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
        success_score: Score,
    ):
        """Test that the attack result includes the tracked adversarial chat conversation IDs."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            attack_scoring_config=AttackScoringConfig(objective_scorer=mock_objective_scorer),
        )

        with (
            patch.object(attack._conversation_manager, "update_conversation_state_async") as mock_update,
            patch.object(attack._prompt_normalizer, "send_prompt_async", new_callable=AsyncMock) as mock_send,
            patch.object(Scorer, "score_response_async", new_callable=AsyncMock) as mock_score,
            patch.object(attack, "_generate_next_prompt_async", new_callable=AsyncMock) as mock_generate,
        ):
            mock_update.return_value = ConversationState(
                turn_count=0, last_user_message="", last_assistant_message_scores=[]
            )
            mock_send.return_value = sample_response
            mock_score.return_value = {"objective_scores": [success_score]}
            mock_generate.return_value = "Test prompt"

            # Run setup and attack
            await attack._setup_async(context=basic_context)
            result = await attack._perform_async(context=basic_context)

            # Verify the result includes the adversarial chat conversation IDs
            assert (
                ConversationReference(
                    conversation_id=basic_context.session.adversarial_chat_conversation_id,
                    conversation_type=ConversationType.ADVERSARIAL,
                )
                in result.related_conversations
            )

    @pytest.mark.asyncio
    async def test_adversarial_chat_conversation_id_uniqueness(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that adversarial chat conversation IDs are unique when added to the set."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat),
            attack_scoring_config=AttackScoringConfig(objective_scorer=mock_objective_scorer),
        )

        # Mock the conversation manager
        with patch.object(attack._conversation_manager, "update_conversation_state_async") as mock_update:
            mock_update.return_value = ConversationState(
                turn_count=0, last_user_message="", last_assistant_message_scores=[]
            )

            # Run setup
            await attack._setup_async(context=basic_context)

            # Get the conversation ID
            conversation_id = basic_context.session.adversarial_chat_conversation_id

            # Verify it was added
            assert (
                ConversationReference(
                    conversation_id=conversation_id,
                    conversation_type=ConversationType.ADVERSARIAL,
                )
                in basic_context.related_conversations
            )
            assert len(basic_context.related_conversations) == 1

            # Try to add the same ID again (should not affect the set)
            basic_context.related_conversations.add(
                ConversationReference(
                    conversation_id=conversation_id,
                    conversation_type=ConversationType.ADVERSARIAL,
                )
            )

            # Verify it's still only one entry
            assert len(basic_context.related_conversations) == 1
