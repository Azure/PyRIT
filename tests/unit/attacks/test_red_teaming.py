# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
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
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import Scorer


@pytest.fixture
def mock_objective_target():
    """Create a mock objective target for testing"""
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = {"id": "mock_target_id"}
    return target


@pytest.fixture
def mock_adversarial_chat():
    """Create a mock adversarial chat target for testing"""
    chat = MagicMock(spec=PromptChatTarget)
    chat.send_prompt_async = AsyncMock()
    chat.set_system_prompt = MagicMock()
    chat.get_identifier.return_value = {"id": "mock_adversarial_chat_id"}
    return chat


@pytest.fixture
def mock_objective_scorer():
    """Create a mock objective scorer for testing"""
    scorer = MagicMock(spec=Scorer)
    scorer.scorer_type = "true_false"
    scorer.score_async = AsyncMock()
    scorer.get_identifier.return_value = {"__type__": "MockScorer"}
    return scorer


@pytest.fixture
def mock_prompt_normalizer():
    """Create a mock prompt normalizer for testing"""
    normalizer = MagicMock(spec=PromptNormalizer)
    normalizer.send_prompt_async = AsyncMock()
    return normalizer


@pytest.fixture
def basic_context():
    """Create a basic multi-turn context for testing"""
    return MultiTurnAttackContext(
        objective="Test objective",
        max_turns=5,
        session=ConversationSession(),
    )


@pytest.fixture
def sample_response():
    """Create a sample response for testing"""
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
        scorer_class_identifier={"__type__": "MockScorer"},
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
        scorer_class_identifier={"__type__": "MockScorer"},
    )


@pytest.fixture
def float_score():
    """Create a float scale score for testing"""
    return Score(
        score_type="float_scale",
        score_value="0.9",
        score_category="test",
        score_value_description="High score",
        score_rationale="Test rationale for high score",
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
        scorer_class_identifier={"__type__": "MockScorer"},
    )


@pytest.mark.usefixtures("patch_central_database")
class TestRedTeamingAttackInitialization:
    """Tests for RedTeamingAttack initialization and configuration"""

    def test_init_with_minimal_required_parameters(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat
    ):
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
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, system_prompt_path
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_system_prompt_path=system_prompt_path,
        )

        assert attack._adversarial_chat_system_prompt_template is not None
        assert attack._adversarial_chat_system_prompt_template.parameters is not None
        assert "objective" in attack._adversarial_chat_system_prompt_template.parameters

    def test_init_with_custom_seed_prompt_string(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat
    ):
        custom_seed = "Custom seed prompt"
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_seed_prompt=custom_seed,
        )

        assert attack._adversarial_chat_seed_prompt.value == custom_seed
        assert attack._adversarial_chat_seed_prompt.data_type == "text"

    def test_init_with_seed_prompt_object(self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat):
        seed_prompt = SeedPrompt(value="Custom seed", data_type="text")
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_seed_prompt=seed_prompt,
        )

        assert attack._adversarial_chat_seed_prompt == seed_prompt

    def test_init_with_invalid_system_prompt_path_raises_error(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat
    ):
        with pytest.raises(FileNotFoundError):
            RedTeamingAttack(
                objective_target=mock_objective_target,
                objective_scorer=mock_objective_scorer,
                adversarial_chat=mock_adversarial_chat,
                adversarial_chat_system_prompt_path="nonexistent_file.yaml",
            )

    def test_init_with_all_custom_configurations(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
    ):
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
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        objective,
        max_turns,
        executed_turns,
        expected_error,
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )
        context = MultiTurnAttackContext(objective=objective, max_turns=max_turns, executed_turns=executed_turns)

        with pytest.raises(ValueError, match=expected_error):
            attack._validate_context(context=context)

    def test_validate_context_with_valid_context(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )
        attack._validate_context(context=basic_context)  # Should not raise


@pytest.mark.usefixtures("patch_central_database")
class TestSetupPhase:
    """Tests for the setup phase of the attack"""

    @pytest.mark.asyncio
    async def test_setup_initializes_achieved_objective_to_false(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Set to True to verify it gets reset
        basic_context.achieved_objective = True

        # Mock conversation manager
        mock_state = ConversationState(turn_count=0)
        attack._conversation_manager.update_conversation_state_async = AsyncMock(return_value=mock_state)

        await attack._setup_async(context=basic_context)

        assert basic_context.achieved_objective is False

    @pytest.mark.asyncio
    async def test_setup_creates_session_if_not_exists(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Don't pass session parameter - let it use the default factory
        context = MultiTurnAttackContext(objective="Test objective", max_turns=5)

        # Mock conversation manager
        mock_state = ConversationState(turn_count=0)
        attack._conversation_manager.update_conversation_state_async = AsyncMock(return_value=mock_state)

        await attack._setup_async(context=context)

        assert context.session is not None
        assert isinstance(context.session, ConversationSession)

    @pytest.mark.asyncio
    async def test_setup_updates_turn_count_from_prepended_conversation(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Mock conversation state with existing turns
        mock_state = ConversationState(turn_count=3)
        attack._conversation_manager.update_conversation_state_async = AsyncMock(return_value=mock_state)

        await attack._setup_async(context=basic_context)

        assert basic_context.executed_turns == 3

    @pytest.mark.asyncio
    async def test_setup_merges_memory_labels_correctly(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
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
        attack._conversation_manager.update_conversation_state_async = AsyncMock(return_value=mock_state)

        await attack._setup_async(context=basic_context)

        # Context labels should override strategy labels for common keys
        assert basic_context.memory_labels == {
            "strategy_label": "strategy_value",
            "context_label": "context_value",
            "common": "context",
        }

    @pytest.mark.asyncio
    async def test_setup_sets_adversarial_chat_system_prompt(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Mock conversation manager
        mock_state = ConversationState(turn_count=0)
        attack._conversation_manager.update_conversation_state_async = AsyncMock(return_value=mock_state)

        await attack._setup_async(context=basic_context)

        # Verify system prompt was set
        mock_adversarial_chat.set_system_prompt.assert_called_once()
        call_args = mock_adversarial_chat.set_system_prompt.call_args
        assert "Test objective" in call_args.kwargs["system_prompt"]
        assert call_args.kwargs["conversation_id"] == basic_context.session.adversarial_chat_conversation_id

    @pytest.mark.asyncio
    async def test_setup_detects_custom_prompt(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Mock conversation state with custom prompt
        mock_state = ConversationState(
            turn_count=0, last_user_message="Custom initial prompt", last_assistant_message_scores=[]
        )
        attack._conversation_manager.update_conversation_state_async = AsyncMock(return_value=mock_state)

        await attack._setup_async(context=basic_context)

        assert basic_context.custom_prompt == "Custom initial prompt"

    @pytest.mark.asyncio
    async def test_setup_retrieves_last_score_matching_scorer_type(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        basic_context,
        success_score,
    ):
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
            scorer_class_identifier={"__type__": "OtherScorer"},
        )

        mock_state = ConversationState(
            turn_count=1,
            last_assistant_message_scores=[other_score, success_score],
        )
        attack._conversation_manager.update_conversation_state_async = AsyncMock(return_value=mock_state)

        await attack._setup_async(context=basic_context)

        assert basic_context.last_score == success_score


@pytest.mark.usefixtures("patch_central_database")
class TestPromptGeneration:
    """Tests for prompt generation logic"""

    @pytest.mark.asyncio
    async def test_generate_next_prompt_uses_custom_prompt_on_first_turn(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.executed_turns = 0
        basic_context.custom_prompt = "Custom first prompt"

        result = await attack._generate_next_prompt(context=basic_context)

        assert result == "Custom first prompt"
        # Should not call adversarial chat
        mock_prompt_normalizer.send_prompt_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_next_prompt_uses_adversarial_chat_after_first_turn(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
        sample_response,
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.executed_turns = 1
        mock_prompt_normalizer.send_prompt_async.return_value = sample_response

        # Mock build_adversarial_prompt
        attack._build_adversarial_prompt = AsyncMock(return_value="Built prompt")

        result = await attack._generate_next_prompt(context=basic_context)

        assert result == "Test response"
        mock_prompt_normalizer.send_prompt_async.assert_called_once()
        attack._build_adversarial_prompt.assert_called_once_with(basic_context)

    @pytest.mark.asyncio
    async def test_generate_next_prompt_raises_on_none_response(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.executed_turns = 1
        mock_prompt_normalizer.send_prompt_async.return_value = None

        # Mock build_adversarial_prompt
        attack._build_adversarial_prompt = AsyncMock(return_value="Built prompt")

        with pytest.raises(ValueError, match="Received no response from adversarial chat"):
            await attack._generate_next_prompt(context=basic_context)


@pytest.mark.usefixtures("patch_central_database")
class TestAdversarialPromptBuilding:
    """Tests for building adversarial prompts"""

    @pytest.mark.asyncio
    async def test_build_adversarial_prompt_returns_seed_when_no_response(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            adversarial_chat_seed_prompt="Initial seed",
        )

        # Mock conversation manager to return no last message
        attack._conversation_manager.get_last_message = MagicMock(return_value=None)

        result = await attack._build_adversarial_prompt(basic_context)

        assert result == "Initial seed"

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
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        basic_context,
        data_type,
        converted_value,
        has_error,
        is_blocked,
        expected_result,
    ):
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
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        basic_context,
        success_score,
    ):
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
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
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
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
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
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        basic_context,
        success_score,
    ):
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
    """Tests for response scoring logic"""

    @pytest.mark.asyncio
    async def test_score_response_successful(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        basic_context,
        sample_response,
        success_score,
    ):
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
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
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
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
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
        "score_value,message_count",
        [
            (True, 0),
            (True, 2),
            (True, 4),
            (False, 0),
            (False, 2),
            (False, 4),
        ],
    )
    @pytest.mark.asyncio
    async def test_score_response_with_various_message_counts(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        basic_context,
        score_value,
        message_count,
    ):
        """Test scoring responses with different conversation lengths"""
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
            scorer_class_identifier={"__type__": "MockScorer"},
        )

        mock_objective_scorer.score_async.return_value = [score]

        # Create conversation history with alternating user/assistant messages
        conversation_history = []
        for i in range(message_count):
            piece = PromptRequestPiece(
                role="user" if i % 2 == 0 else "assistant",
                original_value=f"Message #{i}",
                converted_value=f"Message #{i}",
                original_value_data_type="text",
                converted_value_data_type="text",
            )
            conversation_history.append(piece)

        # Add the response we want to score (assistant message)
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
        self, mock_objective_target, mock_adversarial_chat, basic_context, sample_response
    ):
        """Test that non-boolean scorer types are handled properly"""
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
            scorer_class_identifier={"__type__": "MockScorer"},
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


@pytest.mark.usefixtures("patch_central_database")
class TestAttackExecution:
    """Tests for the main attack execution logic"""

    @pytest.mark.asyncio
    async def test_perform_attack_successful_on_first_turn(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
        sample_response,
        success_score,
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Mock methods
        attack._generate_next_prompt = AsyncMock(return_value="Attack prompt")
        attack._send_prompt_to_target = AsyncMock(return_value=sample_response.request_pieces[0])
        attack._score_response = AsyncMock(return_value=success_score)

        result = await attack._perform_attack_async(context=basic_context)

        assert isinstance(result, AttackResult)
        assert result.achieved_objective is True
        assert result.executed_turns == 1
        assert result.last_response == sample_response.request_pieces[0]
        assert result.last_score == success_score

    @pytest.mark.asyncio
    async def test_perform_attack_reaches_max_turns(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
        sample_response,
        failure_score,
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.max_turns = 3

        # Mock methods to always fail
        attack._generate_next_prompt = AsyncMock(return_value="Attack prompt")
        attack._send_prompt_to_target = AsyncMock(return_value=sample_response.request_pieces[0])
        attack._score_response = AsyncMock(return_value=failure_score)

        result = await attack._perform_attack_async(context=basic_context)

        assert result.achieved_objective is False
        assert result.executed_turns == 3
        assert attack._generate_next_prompt.call_count == 3
        assert attack._send_prompt_to_target.call_count == 3
        assert attack._score_response.call_count == 3

    @pytest.mark.asyncio
    async def test_perform_attack_with_float_scorer_above_threshold(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
        sample_response,
        float_score,
    ):
        mock_objective_scorer.scorer_type = "float_scale"

        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
            attack_scoring_config=AttackScoringConfig(objective_achieved_score_threshold=0.8),
        )

        # Mock methods
        attack._generate_next_prompt = AsyncMock(return_value="Attack prompt")
        attack._send_prompt_to_target = AsyncMock(return_value=sample_response.request_pieces[0])
        attack._score_response = AsyncMock(return_value=float_score)

        result = await attack._perform_attack_async(context=basic_context)

        assert result.achieved_objective is True  # 0.9 > 0.8 threshold

    @pytest.mark.asyncio
    async def test_perform_attack_with_score_as_feedback(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
        sample_response,
        failure_score,
        success_score,
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
            attack_scoring_config=AttackScoringConfig(use_score_as_feedback=True),
        )

        # Mock methods - fail first, succeed second
        attack._generate_next_prompt = AsyncMock(side_effect=["Attack prompt 1", "Attack prompt 2"])
        attack._send_prompt_to_target = AsyncMock(return_value=sample_response.request_pieces[0])
        attack._score_response = AsyncMock(side_effect=[failure_score, success_score])

        result = await attack._perform_attack_async(context=basic_context)

        assert result.achieved_objective is True
        assert result.executed_turns == 2

    @pytest.mark.asyncio
    async def test_perform_attack_raises_on_unexpected_error(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
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


@pytest.mark.usefixtures("patch_central_database")
class TestPromptSending:
    """Tests for sending prompts to target"""

    @pytest.mark.asyncio
    async def test_send_prompt_to_target_successful(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
        sample_response,
    ):
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
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
    ):
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
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
    ):
        """Test sending prompts twice and verify memory state and conversation flow"""
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
        mock_memory_pieces = []

        def mock_get_prompt_request_pieces(conversation_id=None):
            if conversation_id:
                return [p for p in mock_memory_pieces if p.conversation_id == conversation_id]
            return mock_memory_pieces

        # Mock the conversation manager's get_conversation method
        def mock_get_conversation(conversation_id):
            return []  # Return empty list to simulate no previous messages

        attack._conversation_manager._memory.get_prompt_request_pieces = MagicMock(
            side_effect=mock_get_prompt_request_pieces
        )
        attack._conversation_manager._memory.get_conversation = MagicMock(side_effect=mock_get_conversation)
        attack._conversation_manager.get_conversation = MagicMock(side_effect=mock_get_conversation)

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
        grouped_conversations = {}
        for obj in conversations:
            key = obj.conversation_id
            if key in grouped_conversations:
                grouped_conversations[key].append(obj)
            else:
                grouped_conversations[key] = [obj]

        # Should have exactly 2 conversation threads
        assert len(grouped_conversations.keys()) == 2, "Should have target and adversarial chat conversations"


@pytest.mark.usefixtures("patch_central_database")
class TestAttackLifecycle:
    """Tests for the complete attack lifecycle (execute_async)"""

    @pytest.mark.asyncio
    async def test_execute_async_successful_lifecycle(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        basic_context,
        sample_response,
        success_score,
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        # Mock all lifecycle methods
        attack._validate_context = MagicMock()
        attack._setup_async = AsyncMock()
        attack._perform_attack_async = AsyncMock(
            return_value=AttackResult(
                conversation_id=basic_context.session.conversation_id,
                objective=basic_context.objective,
                orchestrator_identifier=attack.get_identifier(),
                achieved_objective=True,
                executed_turns=1,
                last_response=sample_response.request_pieces[0],
                last_score=success_score,
            )
        )
        attack._teardown_async = AsyncMock()

        # Execute the complete lifecycle
        result = await attack.execute_async(context=basic_context)

        # Verify result and proper execution order
        assert isinstance(result, AttackResult)
        assert result.achieved_objective is True
        attack._validate_context.assert_called_once_with(context=basic_context)
        attack._setup_async.assert_called_once_with(context=basic_context)
        attack._perform_attack_async.assert_called_once_with(context=basic_context)
        attack._teardown_async.assert_called_once_with(context=basic_context)

    @pytest.mark.asyncio
    async def test_execute_async_validation_failure_prevents_execution(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

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
    async def test_teardown_async_is_noop(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
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
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        basic_context,
        sample_response,
        failure_score,
        max_turns,
    ):
        """Test that attack respects max_turns configuration"""
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
        attack._conversation_manager.update_conversation_state_async = AsyncMock(return_value=mock_state)

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
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        basic_context,
        sample_response,
        success_score,
    ):
        """Test that memory labels are properly handled during execution"""
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
        attack._conversation_manager.get_conversation = MagicMock(return_value=[])
        attack._conversation_manager._memory.get_conversation = MagicMock(return_value=[])

        with patch.object(attack._prompt_normalizer, "send_prompt_async", return_value=sample_response) as mock_send:
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
    """Tests for edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_score_response_with_empty_scores_list(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context, sample_response
    ):
        """Test handling when scorer returns empty list"""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        mock_objective_scorer.score_async.return_value = []  # Empty scores

        result = await attack._score_response(context=basic_context, response=sample_response.request_pieces[0])

        assert result is None

    @pytest.mark.asyncio
    async def test_build_adversarial_prompt_with_multimodal_response(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat, basic_context
    ):
        """Test handling multimodal responses (e.g., audio, video)"""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            attack_scoring_config=AttackScoringConfig(use_score_as_feedback=False),
        )

        # Mock audio response
        response = MagicMock(spec=PromptRequestPiece)
        response.converted_value_data_type = "audio_path"
        response.has_error.return_value = False

        with pytest.raises(ValueError, match="use_score_as_feedback flag is set to False"):
            attack._handle_adversarial_file_response(response=response, context=basic_context)

    @pytest.mark.asyncio
    async def test_perform_attack_with_alternating_success_failure(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        mock_prompt_normalizer,
        basic_context,
        sample_response,
        success_score,
        failure_score,
    ):
        """Test attack behavior with alternating success/failure scores"""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.max_turns = 5

        # Mock alternating scores
        attack._generate_next_prompt = AsyncMock(return_value="Attack prompt")
        attack._send_prompt_to_target = AsyncMock(return_value=sample_response.request_pieces[0])
        attack._score_response = AsyncMock(
            side_effect=[failure_score, success_score, failure_score, success_score, failure_score]
        )

        result = await attack._perform_attack_async(context=basic_context)

        # Should succeed on turn 2
        assert result.achieved_objective is True
        assert result.executed_turns == 2

    @pytest.mark.asyncio
    async def test_attack_with_prepended_conversation_affects_behavior(
        self,
        mock_objective_target,
        mock_objective_scorer,
        mock_adversarial_chat,
        sample_response,
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

        # Mock conversation state with prepended turns
        mock_state = ConversationState(turn_count=2)  # 2 turns from prepended
        attack._conversation_manager.update_conversation_state_async = AsyncMock(return_value=mock_state)

        await attack._setup_async(context=context)

        assert context.executed_turns == 2  # Should start from turn 3

    @pytest.mark.asyncio
    async def test_concurrent_context_isolation(
        self, mock_objective_target, mock_objective_scorer, mock_adversarial_chat
    ):
        """Test that concurrent contexts don't interfere with each other"""
        attack = RedTeamingAttack(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
            adversarial_chat=mock_adversarial_chat,
        )

        context1 = MultiTurnAttackContext(objective="Objective 1", max_turns=5)
        context2 = MultiTurnAttackContext(objective="Objective 2", max_turns=3)

        # Mock conversation manager to avoid memory access issues
        mock_state = ConversationState(turn_count=0)
        attack._conversation_manager.update_conversation_state_async = AsyncMock(return_value=mock_state)

        # Setup both contexts
        await attack._setup_async(context=context1)
        await attack._setup_async(context=context2)

        # Verify that contexts maintain separate conversation IDs
        assert context1.session.conversation_id != context2.session.conversation_id
        assert context1.session.adversarial_chat_conversation_id != context2.session.adversarial_chat_conversation_id

        # Modify context1 after setup
        context1.executed_turns = 2
        context1.achieved_objective = True

        # Verify context2 is unaffected
        assert context2.executed_turns == 0
        assert context2.achieved_objective == False

        # Also test duplicate() method
        context1_copy = context1.duplicate()
        context2_copy = context2.duplicate()

        # Verify copies are independent
        assert context1_copy.executed_turns == 0  # duplicate() should reset these
        assert context1_copy.achieved_objective == False
        assert context2_copy.executed_turns == 0
        assert context2_copy.achieved_objective == False
