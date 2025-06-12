# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from pathlib import Path
from typing import Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.attacks.base.attack_config import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
)
from pyrit.attacks.base.attack_context import (
    ConversationSession,
    MultiTurnAttackContext,
)
from pyrit.attacks.base.attack_result import AttackOutcome, AttackResult
from pyrit.attacks.components.conversation_manager import ConversationState
from pyrit.attacks.multi_turn.red_teaming import RedTeamingAttack, RTOSystemPromptPaths
from pyrit.exceptions.exception_classes import (
    AttackValidationException,
)
from pyrit.models import (
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
        max_turns=5,
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

    @pytest.mark.parametrize(
        "system_prompt_path",
        [
            RTOSystemPromptPaths.TEXT_GENERATION.value,
            RTOSystemPromptPaths.IMAGE_GENERATION.value,
            RTOSystemPromptPaths.NAIVE_CRESCENDO.value,
            RTOSystemPromptPaths.VIOLENT_DURIAN.value,
            RTOSystemPromptPaths.CRUCIBLE.value,
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
        )

        assert attack._request_converters == converter_config.request_converters
        assert attack._response_converters == converter_config.response_converters
        assert attack._prompt_normalizer == mock_prompt_normalizer

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
class TestContextValidation:
    """Tests for context validation logic"""

    @pytest.mark.parametrize(
        "objective,max_turns,executed_turns,expected_error",
        [
            ("", 5, 0, "Attack objective must be provided"),
            ("Test objective", 0, 0, "Max turns must be positive"),
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
        )
        context = MultiTurnAttackContext(objective=objective, max_turns=max_turns, executed_turns=executed_turns)

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
        )
        attack._validate_context(context=basic_context)  # Should not raise


@pytest.mark.usefixtures("patch_central_database")
class TestSetupPhase:
    """Tests for the setup phase of the attack."""

    @pytest.mark.parametrize(
        "initial_value,expected_value",
        [
            (True, False),  # Test it gets reset
            (False, False),  # Test it stays false
        ],
    )
    @pytest.mark.asyncio
    async def test_setup_initializes_achieved_objective(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        initial_value: bool,
        expected_value: bool,
    ):
        """Test that setup correctly initializes the achieved_objective flag."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.achieved_objective = initial_value

        # Mock conversation manager
        mock_state = ConversationState(turn_count=0)
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=basic_context)

        assert basic_context.achieved_objective == expected_value

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
        """Test that custom prompt is used on first turn when provided."""
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

        result = await attack._generate_next_prompt(context=basic_context)

        assert result == first_prompt
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
        """Test that adversarial chat is used to generate prompts after first turn."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.executed_turns = 1
        mock_prompt_normalizer.send_prompt_async.return_value = sample_response

        # Mock build_adversarial_prompt
        with patch.object(attack, "_build_adversarial_prompt", new_callable=AsyncMock, return_value="Built prompt"):
            result = await attack._generate_next_prompt(context=basic_context)

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
                await attack._generate_next_prompt(context=basic_context)


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

        # Mock conversation manager to return no last message
        with patch.object(attack._conversation_manager, "get_last_message", return_value=None):
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

        response = MagicMock(spec=PromptRequestPiece)
        response.converted_value_data_type = data_type
        response.converted_value = converted_value
        response.has_error.return_value = has_error
        response.is_blocked.return_value = is_blocked
        response.response_error = "Error message"

        basic_context.last_score = None

        result = attack._handle_adversarial_text_response(response=response, context=basic_context)

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

        response = MagicMock(spec=PromptRequestPiece)
        response.converted_value_data_type = "text"
        response.converted_value = "Target response"
        response.has_error.return_value = False

        basic_context.last_score = success_score

        result = attack._handle_adversarial_text_response(response=response, context=basic_context)

        assert "Target response" in result
        assert success_score.score_rationale in result

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

        response = MagicMock(spec=PromptRequestPiece)
        response.converted_value_data_type = "image_path"
        response.has_error.return_value = True
        response.response_error = "File error"

        with pytest.raises(RuntimeError, match="Request to target failed.*File error"):
            attack._handle_adversarial_file_response(response=response, context=basic_context)

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

        response = MagicMock(spec=PromptRequestPiece)
        response.converted_value_data_type = "image_path"
        response.has_error.return_value = False

        with pytest.raises(ValueError, match="use_score_as_feedback flag is set to False"):
            attack._handle_adversarial_file_response(response=response, context=basic_context)

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

        response = MagicMock(spec=PromptRequestPiece)
        response.converted_value_data_type = "image_path"
        response.has_error.return_value = False

        basic_context.last_score = success_score

        result = attack._handle_adversarial_file_response(response=response, context=basic_context)

        assert result == success_score.score_rationale


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

        response_piece = sample_response.request_pieces[0]
        mock_objective_scorer.score_async.return_value = [success_score]

        result = await attack._score_response_async(context=basic_context, response=response_piece)

        assert result == success_score
        mock_objective_scorer.score_async.assert_called_once_with(
            request_response=response_piece, task=basic_context.objective
        )

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
        response_piece.has_error.return_value = True
        response_piece.is_blocked.return_value = True

        result = await attack._score_response_async(context=basic_context, response=response_piece)

        assert result is None
        mock_objective_scorer.score_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_score_response_raises_on_non_blocked_error(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that non-blocked errors raise RuntimeError."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        response_piece = MagicMock(spec=PromptRequestPiece)
        response_piece.has_error.return_value = True
        response_piece.is_blocked.return_value = False
        response_piece.response_error = "Some error"

        with pytest.raises(RuntimeError, match="Response error: Some error"):
            await attack._score_response_async(context=basic_context, response=response_piece)


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
        with patch.object(attack, "_generate_next_prompt", new_callable=AsyncMock, return_value="Attack prompt"):
            with patch.object(
                attack,
                "_send_prompt_to_objective_target_async",
                new_callable=AsyncMock,
                return_value=sample_response.request_pieces[0],
            ):
                with patch.object(attack, "_score_response_async", new_callable=AsyncMock, return_value=score):
                    result = await attack._perform_attack_async(context=basic_context)

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
        )

        basic_context.max_turns = 3

        # Mock methods to always fail
        with patch.object(
            attack, "_generate_next_prompt", new_callable=AsyncMock, return_value="Attack prompt"
        ) as mock_generate:
            with patch.object(
                attack,
                "_send_prompt_to_objective_target_async",
                new_callable=AsyncMock,
                return_value=sample_response.request_pieces[0],
            ) as mock_send:
                with patch.object(
                    attack, "_score_response_async", new_callable=AsyncMock, return_value=failure_score
                ) as mock_score:
                    result = await attack._perform_attack_async(context=basic_context)

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
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
        success_score: Score,
    ):
        """Test successful execution of complete attack lifecycle."""
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
                with patch.object(attack, "_perform_attack_async", new_callable=AsyncMock) as mock_perform:
                    with patch.object(attack, "_teardown_async", new_callable=AsyncMock):
                        # Configure the return value for _perform_attack_async
                        mock_perform.return_value = AttackResult(
                            conversation_id=basic_context.session.conversation_id,
                            objective=basic_context.objective,
                            attack_identifier=attack.get_identifier(),
                            outcome=AttackOutcome.SUCCESS,
                            executed_turns=1,
                            last_response=sample_response.request_pieces[0],
                            last_score=success_score,
                        )

                        # Execute the complete lifecycle
                        result = await attack.execute_async(context=basic_context)

        # Verify result and proper execution order
        assert isinstance(result, AttackResult)
        assert result.outcome == AttackOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_async_validation_failure_prevents_execution(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that validation failure prevents attack execution."""
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
                with patch.object(attack, "_perform_attack_async", new_callable=AsyncMock) as mock_perform:
                    with patch.object(attack, "_teardown_async", new_callable=AsyncMock) as mock_teardown:
                        # Should raise AttackValidationException
                        with pytest.raises(AttackValidationException) as exc_info:
                            await attack.execute_async(context=basic_context)

        # Verify error details
        assert "Context validation failed" in str(exc_info.value)

        # Verify only validation was attempted
        mock_validate.assert_called_once_with(context=basic_context)
        mock_setup.assert_not_called()
        mock_perform.assert_not_called()
        mock_teardown.assert_not_called()

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
