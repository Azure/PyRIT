# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from pathlib import Path
from typing import Dict, List, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.attacks.base.config import AttackConverterConfig, AttackScoringConfig
from pyrit.attacks.base.context import ConversationSession, MultiTurnAttackContext
from pyrit.attacks.base.result import AttackResult
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        assert attack._objective_target == mock_objective_target
        assert attack._objective_scorer == mock_objective_scorer
        assert attack._adversarial_chat == mock_adversarial_chat
        assert isinstance(attack._attack_converter_config, AttackConverterConfig)
        assert isinstance(attack._attack_scoring_config, AttackScoringConfig)
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_system_prompt_path=system_prompt_path,
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_seed_prompt=seed_prompt,
        )

        assert attack._adversarial_chat_seed_prompt.value == expected_value
        if expected_type == str:
            assert attack._adversarial_chat_seed_prompt.data_type == "text"

    def test_init_with_invalid_system_prompt_path_raises_error(
        self, mock_objective_target: MagicMock, mock_objective_scorer: MagicMock, mock_adversarial_chat: MagicMock
    ):
        """Test that invalid system prompt path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            RedTeamingAttack(
                objective_target=mock_objective_target,
                objective_scorer=mock_objective_scorer,
                adversarial_chat=mock_adversarial_chat,
                adversarial_chat_system_prompt_path="nonexistent_file.yaml",
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
        scoring_config = AttackScoringConfig(use_score_as_feedback=True)

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            attack_converter_config=converter_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        assert attack._attack_converter_config == converter_config
        assert attack._attack_scoring_config == scoring_config
        assert attack._prompt_normalizer == mock_prompt_normalizer


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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )
        attack._validate_context(context=basic_context)  # Should not raise


@pytest.mark.usefixtures("patch_central_database")
class TestSetupPhase:
    """Tests for the setup phase of the attack.

    The setup phase initializes the attack context, updates conversation state,
    and configures the adversarial chat system prompt.
    """

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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
        """Test that setup correctly retrieves the last score matching the objective scorer type.

        When multiple scores exist from different scorers, the setup should only
        retrieve the score that matches the objective scorer's type to ensure
        consistent evaluation.
        """
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.executed_turns = 1
        mock_prompt_normalizer.send_prompt_async.return_value = sample_response

        # Mock build_adversarial_prompt
        with patch.object(attack, "_build_adversarial_prompt", new_callable=AsyncMock, return_value="Built prompt"):
            result = await attack._generate_next_prompt(context=basic_context)

        assert result == sample_response.get_piece().original_value
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
    """Tests for building adversarial prompts.

    These tests verify the logic for constructing prompts to send to the
    adversarial chat based on the current conversation state and feedback.
    """

    @pytest.mark.asyncio
    async def test_build_adversarial_prompt_returns_seed_when_no_response(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that the seed prompt is returned when no previous response exists.

        On the first turn or when conversation history is empty, the attack
        should use the configured seed prompt to start the conversation.
        """
        seed = "Initial seed"

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_seed_prompt=seed,
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
        """Test that scoring feedback is appended to text responses when enabled.

        When use_score_as_feedback is True, the scorer's rationale should be
        included with the target response to guide the adversarial chat.
        """
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            attack_scoring_config=AttackScoringConfig(use_score_as_feedback=True),
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            attack_scoring_config=AttackScoringConfig(use_score_as_feedback=False),
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            attack_scoring_config=AttackScoringConfig(use_score_as_feedback=True),
        )

        response = MagicMock(spec=PromptRequestPiece)
        response.converted_value_data_type = "image_path"
        response.has_error.return_value = False

        basic_context.last_score = success_score

        result = attack._handle_adversarial_file_response(response=response, context=basic_context)

        assert result == success_score.score_rationale


@pytest.mark.usefixtures("patch_central_database")
class TestResponseScoring:
    """Tests for response scoring logic.

    These tests verify that responses from the objective target are correctly
    evaluated by the scorer to determine if the attack objective is achieved.
    """

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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        response_piece = sample_response.request_pieces[0]
        mock_objective_scorer.score_async.return_value = [success_score]

        result = await attack._score_response(context=basic_context, response=response_piece)

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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        response_piece = MagicMock(spec=PromptRequestPiece)
        response_piece.has_error.return_value = True
        response_piece.is_blocked.return_value = True

        result = await attack._score_response(context=basic_context, response=response_piece)

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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        response_piece = MagicMock(spec=PromptRequestPiece)
        response_piece.has_error.return_value = True
        response_piece.is_blocked.return_value = False
        response_piece.response_error = "Some error"

        with pytest.raises(RuntimeError, match="Response error: Some error"):
            await attack._score_response(context=basic_context, response=response_piece)

    @pytest.mark.parametrize(
        "score_value",
        [True, False],
    )
    @pytest.mark.asyncio
    async def test_score_response_with_boolean_values(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        score_value: bool,
    ):
        """Test scoring responses with different boolean values."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Create a score with the specified value
        score = Score(
            score_type="true_false",
            score_value=str(score_value).lower(),
            score_category="test",
            score_value_description=f"Test score: {score_value}",
            score_rationale=f"Test rationale for {score_value}",
            score_metadata="{}",
            prompt_request_response_id=str(uuid.uuid4()),
            scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
        )

        mock_objective_scorer.score_async.return_value = [score]

        # Create the response we want to score (assistant message)
        response_to_score = PromptRequestPiece(
            role="assistant",
            original_value="Response to score",
            converted_value="Response to score",
            original_value_data_type="text",
            converted_value_data_type="text",
        )

        # Score the response
        result = await attack._score_response(context=basic_context, response=response_to_score)

        assert result is not None
        assert result.get_value() == score_value
        assert mock_objective_scorer.score_async.called

    @pytest.mark.asyncio
    async def test_score_response_with_non_bool_scorer_type(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
    ):
        """Test that non-boolean scorer types are handled properly."""
        # Create a scorer that returns float_scale instead of true_false
        mock_scorer = MagicMock(spec=Scorer)
        mock_scorer.scorer_type = "float_scale"

        score = Score(
            score_type="float_scale",
            score_value="0.5",
            score_category="test",
            score_value_description="Float scale score",
            score_rationale="Test rationale",
            score_metadata="{}",
            prompt_request_response_id=str(uuid.uuid4()),
            scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
        )

        mock_scorer.score_async.return_value = [score]

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # This should work fine - the attack handles float scores
        result = await attack._score_response(context=basic_context, response=sample_response.request_pieces[0])

        assert result is not None
        assert result.score_value == "0.5"

    @pytest.mark.asyncio
    async def test_score_response_with_empty_scores_list(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
    ):
        """Test handling when scorer returns an empty list of scores.

        Some scorers may return empty lists for certain inputs. The attack
        should handle this gracefully by returning None.
        """
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        mock_objective_scorer.score_async.return_value = []  # Empty scores

        result = await attack._score_response(context=basic_context, response=sample_response.request_pieces[0])

        assert result is None


@pytest.mark.usefixtures("patch_central_database")
class TestAttackExecution:
    """Tests for the main attack execution logic.

    These tests verify the core attack loop that generates prompts,
    sends them to targets, scores responses, and determines when
    the objective is achieved.
    """

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

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
            attack_scoring_config=AttackScoringConfig(objective_achieved_score_threshold=threshold),
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
                attack, "_send_prompt_to_target", new_callable=AsyncMock, return_value=sample_response.request_pieces[0]
            ):
                with patch.object(attack, "_score_response", new_callable=AsyncMock, return_value=score):
                    result = await attack._perform_attack_async(context=basic_context)

        assert result.achieved_objective == expected_achieved

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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.max_turns = 3

        # Mock methods to always fail
        with patch.object(
            attack, "_generate_next_prompt", new_callable=AsyncMock, return_value="Attack prompt"
        ) as mock_generate:
            with patch.object(
                attack, "_send_prompt_to_target", new_callable=AsyncMock, return_value=sample_response.request_pieces[0]
            ) as mock_send:
                with patch.object(
                    attack, "_score_response", new_callable=AsyncMock, return_value=failure_score
                ) as mock_score:
                    result = await attack._perform_attack_async(context=basic_context)

        assert result.achieved_objective is False
        assert result.executed_turns == 3
        assert mock_generate.call_count == 3
        assert mock_send.call_count == 3
        assert mock_score.call_count == 3

    @pytest.mark.asyncio
    async def test_perform_attack_with_score_as_feedback(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
        failure_score: Score,
        success_score: Score,
    ):
        """Test attack execution with score feedback enabled."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
            attack_scoring_config=AttackScoringConfig(use_score_as_feedback=True),
        )

        # Mock methods - fail first, succeed second
        with patch.object(
            attack, "_generate_next_prompt", new_callable=AsyncMock, side_effect=["Attack prompt 1", "Attack prompt 2"]
        ):
            with patch.object(
                attack, "_send_prompt_to_target", new_callable=AsyncMock, return_value=sample_response.request_pieces[0]
            ):
                with patch.object(
                    attack, "_score_response", new_callable=AsyncMock, side_effect=[failure_score, success_score]
                ):
                    result = await attack._perform_attack_async(context=basic_context)

        assert result.achieved_objective is True
        assert result.executed_turns == 2

    @pytest.mark.asyncio
    async def test_perform_attack_raises_on_unexpected_error(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that unexpected errors are properly raised during attack execution"""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Create an error response that's not blocked
        error_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value="",
                    converted_value="",
                    original_value_data_type="text",
                    converted_value_data_type="text",
                    response_error="unknown",
                )
            ]
        )

        normal_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value="Normal response",
                    converted_value="Normal response",
                    original_value_data_type="text",
                    converted_value_data_type="text",
                )
            ]
        )

        # Mock responses: seed prompt response, then error
        mock_prompt_normalizer.send_prompt_async.side_effect = [
            normal_response,  # Seed prompt response
            error_response,  # Error on first attempt
        ]

        await attack._setup_async(context=basic_context)

        with pytest.raises(RuntimeError, match="Response error: unknown"):
            await attack._perform_attack_async(context=basic_context)

    @pytest.mark.asyncio
    async def test_perform_attack_handles_none_score(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
    ):
        """Test that attack handles None scores gracefully."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Mock methods - score returns None (blocked response)
        with patch.object(attack, "_generate_next_prompt", new_callable=AsyncMock, return_value="Attack prompt"):
            with patch.object(
                attack, "_send_prompt_to_target", new_callable=AsyncMock, return_value=sample_response.request_pieces[0]
            ):
                with patch.object(attack, "_score_response", new_callable=AsyncMock, return_value=None):
                    result = await attack._perform_attack_async(context=basic_context)

        # Should complete all turns since None score means not achieved
        assert result.achieved_objective is False
        assert result.executed_turns == basic_context.max_turns
        assert result.last_score is None


@pytest.mark.usefixtures("patch_central_database")
class TestPromptSending:
    """Tests for sending prompts to target.

    These tests verify that prompts are correctly sent to the objective
    target through the prompt normalizer with proper configuration.
    """

    @pytest.mark.asyncio
    async def test_send_prompt_to_target_successful(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
    ):
        """Test successful sending of prompt to target."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        mock_prompt_normalizer.send_prompt_async.return_value = sample_response

        result = await attack._send_prompt_to_target(context=basic_context, prompt="Test prompt")

        assert result == sample_response.request_pieces[0]

        # Verify correct parameters
        call_args = mock_prompt_normalizer.send_prompt_async.call_args
        seed_prompt_group = call_args.kwargs["seed_prompt_group"]
        assert len(seed_prompt_group.prompts) == 1
        assert seed_prompt_group.prompts[0].value == "Test prompt"
        assert call_args.kwargs["conversation_id"] == basic_context.session.conversation_id
        assert call_args.kwargs["target"] == mock_objective_target

    @pytest.mark.asyncio
    async def test_send_prompt_to_target_raises_on_none_response(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that ValueError is raised when target returns None."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        mock_prompt_normalizer.send_prompt_async.return_value = None

        with pytest.raises(ValueError, match="Received no response from the target system"):
            await attack._send_prompt_to_target(context=basic_context, prompt="Test prompt")

    @pytest.mark.asyncio
    async def test_send_prompt_twice_verifies_memory_and_conversation_flow(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test sending prompts twice and verify memory state and conversation flow.

        This unit test verifies that:
        1. Adversarial chat generates prompts based on previous responses
        2. Two separate conversation threads are maintained (adversarial and objective)
        3. Memory correctly tracks all interactions
        """
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Mock responses for the two turns
        first_adversarial_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value="First adversarial response",
                    converted_value="First adversarial response",
                    original_value_data_type="text",
                    converted_value_data_type="text",
                )
            ]
        )

        first_objective_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value="First objective response",
                    converted_value="First objective response",
                    original_value_data_type="text",
                    converted_value_data_type="text",
                )
            ]
        )

        second_adversarial_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value="Second adversarial response",
                    converted_value="Second adversarial response",
                    original_value_data_type="text",
                    converted_value_data_type="text",
                )
            ]
        )

        second_objective_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value="Second objective response",
                    converted_value="Second objective response",
                    original_value_data_type="text",
                    converted_value_data_type="text",
                )
            ]
        )

        # Setup the mock normalizer to return the responses in order
        mock_prompt_normalizer.send_prompt_async.side_effect = [
            first_adversarial_response,
            first_objective_response,
            second_adversarial_response,
            second_objective_response,
        ]

        # Mock the memory to track conversations
        mock_memory_pieces: List[MagicMock] = []

        def mock_get_prompt_request_pieces(**kwargs):
            conversation_id = kwargs.get("conversation_id")
            if conversation_id:
                return [p for p in mock_memory_pieces if p.conversation_id == conversation_id]
            return mock_memory_pieces

        # Mock the conversation manager's get_conversation method
        def mock_get_conversation(conversation_id):
            return []  # Return empty list to simulate no previous messages

        # Use patch to mock the memory methods
        with patch.object(
            attack._conversation_manager._memory,
            "get_prompt_request_pieces",
            side_effect=mock_get_prompt_request_pieces,
        ):
            with patch.object(
                attack._conversation_manager._memory, "get_conversation", side_effect=mock_get_conversation
            ):
                with patch.object(attack._conversation_manager, "get_conversation", side_effect=mock_get_conversation):
                    # Setup attack for execution
                    await attack._setup_async(context=basic_context)

                    # First turn
                    prompt1 = await attack._generate_next_prompt(context=basic_context)
                    response1 = await attack._send_prompt_to_target(context=basic_context, prompt=prompt1)

                    assert response1.converted_value == "First objective response"

                    # Verify the first prompt sent to objective target came from adversarial chat
                    objective_calls = [
                        call
                        for call in mock_prompt_normalizer.send_prompt_async.call_args_list
                        if call.kwargs.get("target") == mock_objective_target
                    ]
                    assert len(objective_calls) == 1
                    first_objective_prompt = objective_calls[0].kwargs["seed_prompt_group"].prompts[0].value
                    assert first_objective_prompt == "First adversarial response"

                    # Update context for second turn
                    basic_context.executed_turns = 1
                    basic_context.last_response = response1

                    # Second turn
                    prompt2 = await attack._generate_next_prompt(context=basic_context)
                    response2 = await attack._send_prompt_to_target(context=basic_context, prompt=prompt2)

                    assert response2.converted_value == "Second objective response"

                    # Verify we have 4 calls total (2 to adversarial, 2 to objective)
                    assert mock_prompt_normalizer.send_prompt_async.call_count == 4

                    # Add mock conversation pieces to simulate memory tracking
                    mock_memory_pieces.extend(
                        [
                            MagicMock(conversation_id=basic_context.session.conversation_id),
                            MagicMock(conversation_id=basic_context.session.conversation_id),
                            MagicMock(conversation_id=basic_context.session.adversarial_chat_conversation_id),
                            MagicMock(conversation_id=basic_context.session.adversarial_chat_conversation_id),
                        ]
                    )

                    # Check memory via conversation manager
                    conversations = attack._conversation_manager._memory.get_prompt_request_pieces()

                    # Group by conversation ID to verify two separate conversations
                    grouped_conversations: Dict[str, List[Union[MagicMock, PromptRequestPiece]]] = {}
                    for obj in conversations:
                        key = obj.conversation_id
                        if key in grouped_conversations:
                            grouped_conversations[key].append(obj)
                        else:
                            grouped_conversations[key] = [obj]

                    # Should have exactly 2 conversation threads
                    assert (
                        len(grouped_conversations.keys()) == 2
                    ), "Should have target and adversarial chat conversations"


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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
                            achieved_objective=True,
                            executed_turns=1,
                            last_response=sample_response.request_pieces[0],
                            last_score=success_score,
                        )

                        # Execute the complete lifecycle
                        result = await attack.execute_async(context=basic_context)

        # Verify result and proper execution order
        assert isinstance(result, AttackResult)
        assert result.achieved_objective is True

    @pytest.mark.asyncio
    async def test_execute_async_validation_failure_prevents_execution(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that validation failure prevents attack execution."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
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
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Should complete without error
        await attack._teardown_async(context=basic_context)
        # No assertions needed - we just want to ensure it runs without exceptions

    @pytest.mark.parametrize("max_turns", [1, 3, 5])
    @pytest.mark.asyncio
    async def test_execute_async_respects_max_turns(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
        failure_score: Score,
        max_turns: int,
    ):
        """Test that attack respects max_turns configuration."""
        basic_context.max_turns = max_turns
        basic_context.executed_turns = 0  # Reset executed turns

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Mock components to always return failure (so we hit max turns)
        mock_objective_scorer.score_async.return_value = [failure_score]

        # Mock the conversation manager to return clean state
        mock_state = ConversationState(turn_count=0)

        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            with patch.object(attack, "_generate_next_prompt", return_value="test prompt") as mock_generate:
                with patch.object(
                    attack, "_send_prompt_to_target", return_value=sample_response.request_pieces[0]
                ) as mock_send:
                    with patch.object(attack, "_score_response", return_value=failure_score) as mock_score:

                        result = await attack.execute_async(context=basic_context)

                        assert result.executed_turns == max_turns
                        assert not result.achieved_objective
                        assert mock_generate.call_count == max_turns
                        assert mock_send.call_count == max_turns
                        assert mock_score.call_count == max_turns

    @pytest.mark.asyncio
    async def test_execute_async_with_memory_labels(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
        success_score: Score,
    ):
        """Test that memory labels are properly handled during execution."""
        # Set up memory labels
        attack_labels = {"attack_id": "test_attack"}
        context_labels = {"username": "test_user", "session": "test_session"}

        basic_context.memory_labels = context_labels

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )
        attack._memory_labels = attack_labels

        mock_objective_scorer.score_async.return_value = [success_score]

        # Mock the conversation manager to avoid memory access issues
        with patch.object(attack._conversation_manager, "get_conversation", return_value=[]):
            with patch.object(attack._conversation_manager._memory, "get_conversation", return_value=[]):
                with patch.object(
                    attack._prompt_normalizer, "send_prompt_async", return_value=sample_response
                ) as mock_send:
                    with patch.object(attack._adversarial_chat, "set_system_prompt") as mock_set_system:

                        result = await attack.execute_async(context=basic_context)

                        assert result.achieved_objective

                        # Check that memory labels were merged properly
                        expected_labels = {**attack_labels, **context_labels}

                        # Verify labels were passed to set_system_prompt
                        mock_set_system.assert_called_once()
                        assert mock_set_system.call_args.kwargs["labels"] == expected_labels

                        # Verify labels were passed to send_prompt_async
                        for call in mock_send.call_args_list:
                            if "labels" in call.kwargs:
                                assert call.kwargs["labels"] == expected_labels


@pytest.mark.usefixtures("patch_central_database")
class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling.

    These tests verify that the attack handles various error conditions,
    edge cases, and unusual inputs correctly without crashing.
    """

    @pytest.mark.parametrize(
        "data_type,use_feedback",
        [
            ("audio_path", False),
            ("image_path", False),
            ("video_path", False),
        ],
    )
    def test_handle_multimodal_response_without_feedback(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        data_type: str,
        use_feedback: bool,
    ):
        """Test handling multimodal responses without feedback enabled."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            attack_scoring_config=AttackScoringConfig(use_score_as_feedback=use_feedback),
        )

        response = MagicMock(spec=PromptRequestPiece)
        response.converted_value_data_type = data_type
        response.has_error.return_value = False

        with pytest.raises(ValueError, match="use_score_as_feedback flag is set to False"):
            attack._handle_adversarial_file_response(response=response, context=basic_context)

    @pytest.mark.asyncio
    async def test_perform_attack_with_alternating_success_failure(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
        success_score: Score,
        failure_score: Score,
    ):
        """Test attack behavior with alternating success/failure scores."""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.max_turns = 5

        # Mock alternating scores using patch.object
        with patch.object(attack, "_generate_next_prompt", new_callable=AsyncMock, return_value="Attack prompt"):
            with patch.object(
                attack, "_send_prompt_to_target", new_callable=AsyncMock, return_value=sample_response.request_pieces[0]
            ):
                with patch.object(
                    attack,
                    "_score_response",
                    new_callable=AsyncMock,
                    side_effect=[failure_score, success_score, failure_score, success_score, failure_score],
                ):
                    result = await attack._perform_attack_async(context=basic_context)

        # Should succeed on turn 2
        assert result.achieved_objective is True
        assert result.executed_turns == 2

    @pytest.mark.asyncio
    async def test_attack_with_prepended_conversation_affects_behavior(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        sample_response: PromptRequestResponse,
    ):
        """Test that prepended conversation properly affects attack flow"""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Create prepended conversation
        prepended_conv = [
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        original_value="Previous prompt",
                        converted_value="Previous prompt",
                        original_value_data_type="text",
                        converted_value_data_type="text",
                    )
                ]
            ),
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="assistant",
                        original_value="Previous response",
                        converted_value="Previous response",
                        original_value_data_type="text",
                        converted_value_data_type="text",
                    )
                ]
            ),
        ]

        context = MultiTurnAttackContext(
            objective="Test objective",
            max_turns=5,
            prepended_conversation=prepended_conv,
        )

        # Mock the conversation manager to return state with turn count
        mock_state = ConversationState(turn_count=2)
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=context)

        assert context.executed_turns == 2  # Should start from turn 3

    @pytest.mark.asyncio
    async def test_setup_raises_on_empty_system_prompt(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
    ):
        """Test that setup raises error when system prompt renders to empty string.

        The adversarial chat requires a non-empty system prompt to function.
        This test ensures proper validation of the rendered prompt.
        """
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Mock the template to render empty string
        with patch.object(attack._adversarial_chat_system_prompt_template, "render_template_value", return_value=""):
            # Mock conversation manager
            mock_state = ConversationState(turn_count=0)
            with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
                with pytest.raises(ValueError, match="Adversarial chat system prompt must be defined"):
                    await attack._setup_async(context=basic_context)

    @pytest.mark.asyncio
    async def test_score_response_with_multiple_scores(
        self,
        mock_objective_target: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: MultiTurnAttackContext,
        sample_response: PromptRequestResponse,
    ):
        """Test that score_response correctly returns the first score when multiple are returned.

        Some scorers may return multiple scores for a single response. The attack
        should use the first score as the primary evaluation result.
        """
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Create multiple scores
        scores = [
            Score(
                score_type="true_false",
                score_value="true",
                score_category="test",
                score_value_description=f"{s} score",
                score_rationale=f"{s} rationale",
                score_metadata="{}",
                prompt_request_response_id=str(uuid.uuid4()),
                scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
            )
            for s in ["First", "Second"]
        ]

        scores[1].score_value = "false"  # Ensure second score is different
        mock_objective_scorer.score_async.return_value = scores

        result = await attack._score_response(context=basic_context, response=sample_response.request_pieces[0])

        # Should return the first score
        assert result is not None
        assert result == scores[0]
        assert result.score_value_description == scores[0].score_value_description
