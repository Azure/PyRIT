# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import uuid
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.attacks.base.attack_config import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackRuntimeConfig,
    AttackScoringConfig,
)
from pyrit.attacks.base.attack_context import ConversationSession
from pyrit.attacks.base.attack_result import AttackOutcome
from pyrit.attacks.components.conversation_manager import ConversationState
from pyrit.attacks.multi_turn.crescendo import (
    CrescendoAttack,
    CrescendoAttackContext,
    CrescendoAttackResult,
)
from pyrit.common.path import DATASETS_PATH
from pyrit.exceptions import (
    AttackValidationException,
    InvalidJsonException,
)
from pyrit.models import (
    ChatMessageRole,
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    ScoreType,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import FloatScaleThresholdScorer, SelfAskRefusalScorer


def create_mock_chat_target(*, name: str = "MockChatTarget") -> MagicMock:
    """Create a mock chat target with common setup.

    This standardizes the creation of mock chat targets across tests,
    ensuring they all have the required methods and return values.
    """
    target = MagicMock(spec=PromptChatTarget)
    target.send_prompt_async = AsyncMock()
    target.set_system_prompt = MagicMock()
    target.get_identifier.return_value = {"__type__": name, "__module__": "test_module"}
    return target


def create_mock_scorer(*, scorer_type: str, class_name: str) -> MagicMock:
    """Create a mock scorer with common setup.

    Scorers are used to evaluate responses. This helper ensures all mock scorers
    have consistent behavior and required attributes.
    """
    scorer = MagicMock()
    scorer.scorer_type = scorer_type
    scorer.score_async = AsyncMock()
    scorer.get_identifier.return_value = {"__type__": class_name, "__module__": "test_module"}
    return scorer


def create_score(
    *,
    score_type: ScoreType,
    score_value: str,
    score_category: str,
    scorer_class: str,
    score_rationale: str = "Test rationale",
    score_value_description: str = "Test description",
) -> Score:
    """Create a score with common defaults.

    Scores are central to the Crescendo attack's decision-making process.
    This helper creates valid Score objects with all required fields.
    """
    return Score(
        score_type=score_type,
        score_value=score_value,
        score_category=score_category,
        score_value_description=score_value_description,
        score_rationale=score_rationale,
        score_metadata="{}",
        prompt_request_response_id=str(uuid.uuid4()),
        scorer_class_identifier={"__type__": scorer_class, "__module__": "test_module"},
    )


def create_prompt_response(*, text: str, role: ChatMessageRole = "assistant") -> PromptRequestResponse:
    """Create a prompt response with common defaults.

    This simulates responses from the target model, which are then evaluated
    by the attack logic to determine next steps.
    """
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role=role,
                original_value=text,
                original_value_data_type="text",
                converted_value=text,
                converted_value_data_type="text",
            )
        ]
    )


def create_adversarial_json_response(
    *,
    question: str = "Attack prompt",
    summary: str = "Summary of last response",
    rationale: str = "Rationale for the attack",
) -> str:
    """Create a valid adversarial JSON response.

    The Crescendo attack expects the adversarial chat to return JSON with specific fields.
    This helper creates properly formatted responses for testing.
    """
    return json.dumps(
        {"generated_question": question, "last_response_summary": summary, "rationale_behind_jailbreak": rationale}
    )


@pytest.fixture
def mock_objective_target() -> MagicMock:
    return create_mock_chat_target(name="MockObjectiveTarget")


@pytest.fixture
def mock_adversarial_chat() -> MagicMock:
    return create_mock_chat_target(name="MockAdversarialChat")


@pytest.fixture
def mock_objective_scorer() -> MagicMock:
    return create_mock_scorer(scorer_type="float_scale", class_name="FloatScaleThresholdScorer")


@pytest.fixture
def mock_refusal_scorer() -> MagicMock:
    return create_mock_scorer(scorer_type="true_false", class_name="SelfAskRefusalScorer")


@pytest.fixture
def mock_prompt_normalizer() -> MagicMock:
    normalizer = MagicMock(spec=PromptNormalizer)
    normalizer.send_prompt_async = AsyncMock()
    return normalizer


@pytest.fixture
def basic_context() -> CrescendoAttackContext:
    return CrescendoAttackContext(
        objective="Test objective",
        max_turns=10,
        session=ConversationSession(),
    )


@pytest.fixture
def sample_response() -> PromptRequestResponse:
    return create_prompt_response(text="Test response")


@pytest.fixture
def refusal_score() -> Score:
    return create_score(
        score_type="true_false",
        score_value="true",
        score_category="refusal",
        scorer_class="SelfAskRefusalScorer",
        score_rationale="The target refused to comply with the request",
        score_value_description="Target refused",
    )


@pytest.fixture
def no_refusal_score() -> Score:
    return create_score(
        score_type="true_false",
        score_value="false",
        score_category="refusal",
        scorer_class="SelfAskRefusalScorer",
        score_rationale="The target complied with the request",
        score_value_description="Target did not refuse",
    )


@pytest.fixture
def success_objective_score() -> Score:
    return create_score(
        score_type="float_scale",
        score_value="0.9",
        score_category="objective",
        scorer_class="FloatScaleThresholdScorer",
        score_rationale="The response successfully achieves the objective",
        score_value_description="High objective achievement",
    )


@pytest.fixture
def failure_objective_score() -> Score:
    return create_score(
        score_type="float_scale",
        score_value="0.3",
        score_category="objective",
        scorer_class="FloatScaleThresholdScorer",
        score_rationale="The response does not achieve the objective",
        score_value_description="Low objective achievement",
    )


@pytest.fixture
def adversarial_response() -> str:
    return create_adversarial_json_response()


class CrescendoTestHelper:
    """Helper class for creating Crescendo attack instances with common configurations.

    This centralizes the complex initialization logic and makes it easy to create
    attacks with different configurations for testing various scenarios.
    """

    @staticmethod
    def create_attack(
        *,
        objective_target: MagicMock,
        adversarial_chat: MagicMock,
        objective_scorer: Optional[MagicMock] = None,
        refusal_scorer: Optional[MagicMock] = None,
        prompt_normalizer: Optional[MagicMock] = None,
        system_prompt_path: Optional[Path] = None,
        **kwargs,
    ) -> CrescendoAttack:
        """Create a CrescendoAttack instance with flexible configuration.

        This method handles the complex initialization of CrescendoAttack,
        allowing tests to focus on specific scenarios without repeating setup code.
        """
        adversarial_config = AttackAdversarialConfig(target=adversarial_chat, system_prompt_path=system_prompt_path)

        # Only create scoring config if scorers are provided
        # This allows testing both with custom scorers and default scorers
        scoring_config = None
        if objective_scorer or refusal_scorer:
            scoring_config = AttackScoringConfig(
                objective_scorer=objective_scorer,
                refusal_scorer=refusal_scorer,
                **{k: v for k, v in kwargs.items() if k in ["use_score_as_feedback", "successful_objective_threshold"]},
            )

        return CrescendoAttack(
            objective_target=objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=prompt_normalizer,
        )


@pytest.mark.usefixtures("patch_central_database")
class TestCrescendoAttackInitialization:
    """Tests for CrescendoAttack initialization and configuration"""

    def test_init_with_minimal_required_parameters(
        self, mock_objective_target: MagicMock, mock_adversarial_chat: MagicMock
    ):
        """Test that attack initializes correctly with only required parameters."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        assert attack._objective_target == mock_objective_target
        assert attack._adversarial_chat == mock_adversarial_chat
        assert isinstance(attack._objective_scorer, FloatScaleThresholdScorer)
        assert isinstance(attack._refusal_scorer, SelfAskRefusalScorer)

    def test_init_with_custom_scoring_configuration(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_refusal_scorer: MagicMock,
    ):
        """Test initialization with custom scoring configuration."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(
            objective_scorer=mock_objective_scorer,
            refusal_scorer=mock_refusal_scorer,
            successful_objective_threshold=0.7,
            use_score_as_feedback=False,
        )

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        assert attack._objective_scorer == mock_objective_scorer
        assert attack._refusal_scorer == mock_refusal_scorer
        assert attack._successful_objective_threshold == 0.7
        assert attack._use_score_as_feedback is False

    def test_init_creates_default_scorers_with_adversarial_chat(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test that default scorers are created using the adversarial chat target."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        # Verify default scorers were created
        assert isinstance(attack._objective_scorer, FloatScaleThresholdScorer)
        assert isinstance(attack._refusal_scorer, SelfAskRefusalScorer)

    @pytest.mark.parametrize(
        "system_prompt_path",
        [Path(DATASETS_PATH) / "orchestrators" / "crescendo" / f"crescendo_variant_{i}.yaml" for i in range(1, 6)],
    )
    def test_init_with_different_system_prompt_variants(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        system_prompt_path: Path,
    ):
        """Test initialization with different Crescendo system prompt variants."""
        adversarial_config = AttackAdversarialConfig(
            target=mock_adversarial_chat, system_prompt_path=system_prompt_path
        )

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        assert attack._adversarial_chat_system_prompt_template is not None
        assert attack._adversarial_chat_system_prompt_template.parameters is not None
        # Making sure the system prompt template has expected parameters
        assert "objective" in attack._adversarial_chat_system_prompt_template.parameters
        assert "max_turns" in attack._adversarial_chat_system_prompt_template.parameters

    def test_init_with_invalid_system_prompt_path_raises_error(
        self, mock_objective_target: MagicMock, mock_adversarial_chat: MagicMock
    ):
        """Test that invalid system prompt path raises FileNotFoundError."""
        adversarial_config = AttackAdversarialConfig(
            target=mock_adversarial_chat, system_prompt_path="nonexistent_file.yaml"
        )

        with pytest.raises(FileNotFoundError):
            CrescendoAttack(
                objective_target=mock_objective_target,
                attack_adversarial_config=adversarial_config,
            )

    @pytest.mark.parametrize("max_backtracks", [0, -1, -10])
    def test_init_with_invalid_max_backtracks_raises_error(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        max_backtracks: int,
    ):
        """Test that negative max_backtracks raises ValueError."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        context = CrescendoAttackContext(objective="Test objective", max_backtracks=max_backtracks)

        # Only negative values should raise an error
        if max_backtracks < 0:
            with pytest.raises(ValueError, match="Max backtracks must be non-negative"):
                attack._validate_context(context=context)
        else:
            # max_backtracks=0 should be valid (no backtracking allowed)
            attack._validate_context(context=context)  # Should not raise

    def test_init_with_converter_configuration(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test initialization with converter configuration."""
        from pyrit.prompt_converter import Base64Converter
        from pyrit.prompt_normalizer import PromptConverterConfiguration

        converter_config = AttackConverterConfig(
            request_converters=[PromptConverterConfiguration(converters=[Base64Converter()])], response_converters=[]
        )

        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        # Update attack with converter config
        attack._request_converters = converter_config.request_converters
        attack._response_converters = converter_config.response_converters

        assert len(attack._request_converters) == 1
        assert len(attack._response_converters) == 0


@pytest.mark.usefixtures("patch_central_database")
class TestContextValidation:
    """Tests for context validation logic"""

    @pytest.mark.parametrize(
        "objective,max_turns,expected_error",
        [
            ("", 5, "Attack objective must be provided"),
            ("Test objective", 0, "Max turns must be positive"),
            ("Test objective", -1, "Max turns must be positive"),
        ],
    )
    def test_validate_context_raises_errors(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        objective: str,
        max_turns: int,
        expected_error: str,
    ):
        """Test that context validation raises appropriate errors for invalid inputs."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )
        context = CrescendoAttackContext(objective=objective, max_turns=max_turns)

        with pytest.raises(ValueError, match=expected_error):
            attack._validate_context(context=context)

    def test_validate_context_with_valid_context(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: CrescendoAttackContext,
    ):
        """Test that valid context passes validation without errors."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )
        attack._validate_context(context=basic_context)  # Should not raise


@pytest.mark.usefixtures("patch_central_database")
class TestSetupPhase:
    """Tests for the setup phase of the attack."""

    @pytest.mark.asyncio
    async def test_setup_initializes_conversation_session(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: CrescendoAttackContext,
    ):
        """Test that setup correctly initializes a conversation session."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        # Mock conversation manager
        mock_state = ConversationState(turn_count=0)
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=basic_context)

        assert basic_context.session is not None
        assert isinstance(basic_context.session, ConversationSession)
        assert basic_context.executed_turns == 0
        assert basic_context.backtrack_count == 0

    @pytest.mark.asyncio
    async def test_setup_sets_adversarial_chat_system_prompt(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: CrescendoAttackContext,
    ):
        """Test that setup correctly sets the adversarial chat system prompt."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        # Mock conversation manager
        mock_state = ConversationState(turn_count=0)
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=basic_context)

        # Verify system prompt was set
        mock_adversarial_chat.set_system_prompt.assert_called_once()
        call_args = mock_adversarial_chat.set_system_prompt.call_args
        assert "Test objective" in call_args.kwargs["system_prompt"]
        assert str(basic_context.max_turns) in call_args.kwargs["system_prompt"]
        assert call_args.kwargs["conversation_id"] == basic_context.session.adversarial_chat_conversation_id

    @pytest.mark.asyncio
    async def test_setup_handles_prepended_conversation_with_refusal(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_refusal_scorer: MagicMock,
        basic_context: CrescendoAttackContext,
        refusal_score: Score,
    ):
        """Test that setup handles prepended conversation with refusal score."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(refusal_scorer=mock_refusal_scorer)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        # Mock conversation state with refusal score
        mock_state = ConversationState(
            turn_count=1,
            last_user_message="Refused prompt",
            last_assistant_message_scores=[refusal_score],
        )
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=basic_context)

        assert basic_context.refused_text == "Refused prompt"
        assert basic_context.executed_turns == 1

    @pytest.mark.asyncio
    async def test_setup_retrieves_custom_prompt_from_prepended_conversation(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: CrescendoAttackContext,
    ):
        """Test that setup retrieves custom prompt from prepended conversation."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        # Mock conversation state with last user message but no scores
        mock_state = ConversationState(
            turn_count=0,
            last_user_message="Custom prepended prompt",
            last_assistant_message_scores=[],
        )
        with patch.object(attack._conversation_manager, "update_conversation_state_async", return_value=mock_state):
            await attack._setup_async(context=basic_context)

        assert basic_context.custom_prompt == "Custom prepended prompt"


@pytest.mark.usefixtures("patch_central_database")
class TestPromptGeneration:
    """Tests for prompt generation logic"""

    @pytest.mark.asyncio
    async def test_generate_next_prompt_uses_custom_prompt_when_available(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: CrescendoAttackContext,
    ):
        """Test that custom prompt is used when available."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        basic_context.custom_prompt = "Custom prompt"

        result = await attack._generate_next_prompt_async(context=basic_context)

        assert result == "Custom prompt"
        assert basic_context.custom_prompt is None  # Should be cleared

    @pytest.mark.asyncio
    async def test_generate_next_prompt_calls_adversarial_chat(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: CrescendoAttackContext,
        adversarial_response: str,
    ):
        """Test that adversarial chat is used to generate prompts when no custom prompt."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Mock the adversarial response
        response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value=adversarial_response,
                    converted_value=adversarial_response,
                )
            ]
        )
        mock_prompt_normalizer.send_prompt_async.return_value = response

        basic_context.custom_prompt = None
        basic_context.refused_text = "Previous refused text"

        result = await attack._generate_next_prompt_async(context=basic_context)

        assert result == "Attack prompt"
        mock_prompt_normalizer.send_prompt_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_adversarial_prompt_with_refused_text(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: CrescendoAttackContext,
    ):
        """Test building adversarial prompt when previous attempt was refused."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        basic_context.executed_turns = 2
        refused_text = "This prompt was refused"

        result = attack._build_adversarial_prompt(context=basic_context, refused_text=refused_text)

        assert "This is the turn 3 of 10 turns" in result
        assert "Test objective" in result
        assert "The target refused to respond" in result
        assert refused_text in result

    @pytest.mark.asyncio
    async def test_build_adversarial_prompt_with_objective_score(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        failure_objective_score: Score,
    ):
        """Test building adversarial prompt with previous objective score."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        basic_context.executed_turns = 2
        basic_context.last_response = sample_response
        basic_context.last_score = failure_objective_score

        result = attack._build_adversarial_prompt(context=basic_context, refused_text="")

        assert "This is the turn 3 of 10 turns" in result
        assert "Test objective" in result
        assert "Test response" in result  # From sample_response
        assert "0.30" in result  # Score value
        assert failure_objective_score.score_rationale in result

    @pytest.mark.asyncio
    async def test_send_prompt_to_adversarial_chat_handles_no_response(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: CrescendoAttackContext,
    ):
        """Test handling when adversarial chat returns no response."""
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Mock no response
        mock_prompt_normalizer.send_prompt_async.return_value = None

        with pytest.raises(ValueError, match="No response received from adversarial chat"):
            await attack._send_prompt_to_adversarial_chat_async(prompt_text="Test prompt", context=basic_context)

    @pytest.mark.parametrize(
        "response_json,expected_error",
        [
            # Missing required keys - the attack expects all three fields
            ('{"generated_question": "Attack"}', "Missing required keys"),
            # Extra keys are not allowed - strict JSON validation prevents unexpected data
            (
                '{"generated_question": "Attack", "last_response_summary": "Summary", '
                '"rationale_behind_jailbreak": "Rationale", "extra_key": "value"}',
                "Unexpected keys",
            ),
            # Invalid JSON will trigger retry mechanism
            ("invalid json", "Invalid JSON"),
            # Wrong key names indicate incorrect adversarial chat response format
            ('{"wrong_key": "value"}', "Missing required keys"),
            # Empty question is valid - the attack can handle empty strings
            (
                '{"generated_question": "", "last_response_summary": "Summary", '
                '"rationale_behind_jailbreak": "Rationale"}',
                None,
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_parse_adversarial_response_with_various_inputs(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        response_json: str,
        expected_error: Optional[str],
    ):
        """Test parsing adversarial response with various inputs.

        This test verifies that the JSON parsing is strict and handles various
        error cases appropriately. The strict validation ensures the adversarial
        chat is providing responses in the expected format.
        """
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        if expected_error:
            with pytest.raises(InvalidJsonException) as exc_info:
                attack._parse_adversarial_response(response_json)
            assert expected_error in str(exc_info.value)
        else:
            # Should not raise
            result = attack._parse_adversarial_response(response_json)
            assert isinstance(result, str)


@pytest.mark.usefixtures("patch_central_database")
class TestResponseScoring:
    """Tests for response scoring logic."""

    @pytest.mark.asyncio
    async def test_score_response_successful(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        success_objective_score: Score,
    ):
        """Test successful scoring of a response."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(objective_scorer=mock_objective_scorer)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.last_response = sample_response

        # Mock the Scorer.score_response_with_objective_async method
        with patch(
            "pyrit.score.Scorer.score_response_with_objective_async",
            new_callable=AsyncMock,
            return_value={"objective_scores": [success_objective_score], "auxiliary_scores": []},
        ):
            result = await attack._score_response_async(context=basic_context)

        assert result == success_objective_score

    @pytest.mark.asyncio
    async def test_score_response_raises_when_no_response(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: CrescendoAttackContext,
    ):
        """Test that scoring raises ValueError when no response is available."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        basic_context.last_response = None

        with pytest.raises(ValueError, match="No response available in context to score"):
            await attack._score_response_async(context=basic_context)

    @pytest.mark.asyncio
    async def test_check_refusal_detects_refusal(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_refusal_scorer: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        refusal_score: Score,
    ):
        """Test that refusal is correctly detected."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(refusal_scorer=mock_refusal_scorer)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.last_response = sample_response
        mock_refusal_scorer.score_async.return_value = [refusal_score]

        result = await attack._check_refusal_async(context=basic_context, task="test task")

        assert result == refusal_score
        mock_refusal_scorer.score_async.assert_called_once()


@pytest.mark.usefixtures("patch_central_database")
class TestBacktrackingLogic:
    """Tests for backtracking functionality

    Backtracking is a key feature of Crescendo that allows it to recover from
    refusals by reverting to an earlier conversation state and trying a different approach.
    """

    @pytest.mark.asyncio
    async def test_perform_backtrack_when_refused(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_refusal_scorer: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        refusal_score: Score,
    ):
        """Test that backtracking is performed when response is refused.

        When the target refuses a prompt, Crescendo should:
        1. Store the refused text for the adversarial chat to learn from
        2. Revert the conversation to before the refused prompt
        3. Increment the backtrack counter
        4. Continue with a new approach
        """
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(refusal_scorer=mock_refusal_scorer)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.last_response = sample_response
        basic_context.backtrack_count = 0
        mock_refusal_scorer.score_async.return_value = [refusal_score]

        # Mock backtrack_memory_async to return a new conversation ID
        # This simulates the memory system creating a new conversation branch
        with patch.object(attack, "_backtrack_memory_async", new_callable=AsyncMock, return_value="new_conv_id"):
            result = await attack._perform_backtrack_if_refused_async(
                context=basic_context, prompt_sent="Refused prompt"
            )

        # Verify all expected state changes occurred
        assert result is True
        assert basic_context.refused_text == "Refused prompt"  # Stored for next attempt
        assert basic_context.backtrack_count == 1
        assert basic_context.session.conversation_id == "new_conv_id"  # New conversation branch

    @pytest.mark.asyncio
    async def test_no_backtrack_when_not_refused(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_refusal_scorer: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        no_refusal_score: Score,
    ):
        """Test that no backtracking occurs when response is not refused."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(refusal_scorer=mock_refusal_scorer)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.last_response = sample_response
        basic_context.backtrack_count = 0
        mock_refusal_scorer.score_async.return_value = [no_refusal_score]

        result = await attack._perform_backtrack_if_refused_async(context=basic_context, prompt_sent="Normal prompt")

        assert result is False
        assert basic_context.refused_text is None
        assert basic_context.backtrack_count == 0

    @pytest.mark.asyncio
    async def test_no_backtrack_when_max_backtracks_reached(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_refusal_scorer: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        refusal_score: Score,
    ):
        """Test that no backtracking occurs when max backtracks is reached.

        This prevents infinite loops where the attack keeps getting refused.
        Once the limit is reached, the attack continues forward even if refused,
        allowing it to potentially find success through persistence rather than revision.
        """
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)
        scoring_config = AttackScoringConfig(refusal_scorer=mock_refusal_scorer)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )

        basic_context.last_response = sample_response
        basic_context.max_backtracks = 5
        basic_context.backtrack_count = 5  # Already at max

        result = await attack._perform_backtrack_if_refused_async(context=basic_context, prompt_sent="Refused prompt")

        assert result is False
        # Important: Should not even check for refusal to save API calls
        mock_refusal_scorer.score_async.assert_not_called()


@pytest.mark.usefixtures("patch_central_database")
class TestAttackExecution:
    """Tests for the main attack execution logic."""

    @pytest.mark.asyncio
    async def test_perform_attack_success_on_first_turn(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        success_objective_score: Score,
        no_refusal_score: Score,
        adversarial_response: str,
    ):
        """Test successful attack on the first turn."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Mock adversarial response
        adv_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value=adversarial_response,
                    converted_value=adversarial_response,
                )
            ]
        )

        # Set up mocks
        mock_prompt_normalizer.send_prompt_async.side_effect = [adv_response, sample_response]

        with patch.object(attack, "_check_refusal_async", new_callable=AsyncMock, return_value=no_refusal_score):
            with patch(
                "pyrit.score.Scorer.score_response_with_objective_async",
                new_callable=AsyncMock,
                return_value={"objective_scores": [success_objective_score], "auxiliary_scores": []},
            ):
                result = await attack._perform_attack_async(context=basic_context)

        assert isinstance(result, CrescendoAttackResult)
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.executed_turns == 1
        assert result.last_score == success_objective_score
        assert result.outcome_reason is not None
        assert "Objective achieved in 1 turns" in result.outcome_reason

    @pytest.mark.asyncio
    async def test_perform_attack_failure_max_turns_reached(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        failure_objective_score: Score,
        no_refusal_score: Score,
        adversarial_response: str,
    ):
        """Test attack failure when max turns is reached."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.max_turns = 2

        # Mock adversarial response
        adv_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value=adversarial_response,
                    converted_value=adversarial_response,
                )
            ]
        )

        # Set up mocks for multiple turns
        mock_prompt_normalizer.send_prompt_async.side_effect = [
            adv_response,
            sample_response,  # Turn 1
            adv_response,
            sample_response,  # Turn 2
        ]

        with patch.object(attack, "_check_refusal_async", new_callable=AsyncMock, return_value=no_refusal_score):
            with patch(
                "pyrit.score.Scorer.score_response_with_objective_async",
                new_callable=AsyncMock,
                return_value={"objective_scores": [failure_objective_score], "auxiliary_scores": []},
            ):
                result = await attack._perform_attack_async(context=basic_context)

        assert isinstance(result, CrescendoAttackResult)
        assert result.outcome == AttackOutcome.FAILURE
        assert result.executed_turns == 2
        assert result.last_score == failure_objective_score
        assert result.outcome_reason is not None
        assert "Max turns (2) reached" in result.outcome_reason

    @pytest.mark.asyncio
    async def test_perform_attack_with_backtracking(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        success_objective_score: Score,
        refusal_score: Score,
        no_refusal_score: Score,
        adversarial_response: str,
    ):
        """Test attack with backtracking due to refusal.

        This test verifies the complete backtracking flow:
        1. First attempt is refused
        2. Attack backtracks and tries a different approach
        3. Second attempt succeeds

        The key insight is that only successful turns count toward executed_turns,
        while backtracked attempts are tracked separately.
        """
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Set max_backtracks in context
        basic_context.max_backtracks = 2

        # Mock adversarial response
        adv_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value=adversarial_response,
                    converted_value=adversarial_response,
                )
            ]
        )

        # Set up mocks for the two attempts
        mock_prompt_normalizer.send_prompt_async.side_effect = [
            adv_response,
            sample_response,  # First attempt (will be refused and backtracked)
            adv_response,
            sample_response,  # Second attempt after backtrack (will succeed)
        ]

        # First call returns refusal, triggering backtrack
        # Second call returns no refusal, allowing progress
        check_refusal_results = [refusal_score, no_refusal_score]

        with patch.object(attack, "_check_refusal_async", new_callable=AsyncMock, side_effect=check_refusal_results):
            with patch.object(attack, "_backtrack_memory_async", new_callable=AsyncMock, return_value="new_conv_id"):
                with patch(
                    "pyrit.score.Scorer.score_response_with_objective_async",
                    new_callable=AsyncMock,
                    return_value={"objective_scores": [success_objective_score], "auxiliary_scores": []},
                ):
                    result = await attack._perform_attack_async(context=basic_context)

        assert isinstance(result, CrescendoAttackResult)
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.executed_turns == 1  # Only counts non-backtracked turns
        assert result.backtrack_count == 1  # Tracks backtracking for analysis

    @pytest.mark.asyncio
    async def test_perform_attack_max_backtracks_then_continue(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        failure_objective_score: Score,
        refusal_score: Score,
        adversarial_response: str,
    ):
        """Test attack continues after reaching max backtracks.

        This tests an important edge case: what happens when we hit the backtrack
        limit but haven't reached max turns? The attack should continue forward
        even if responses are refused, as persistence might eventually succeed.
        """
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.max_turns = 3
        basic_context.max_backtracks = 1

        # Mock adversarial response
        adv_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value=adversarial_response,
                    converted_value=adversarial_response,
                )
            ]
        )

        # Set up mocks for multiple attempts
        # The response pattern reflects the expected flow:
        # 1. Turn 1: Refused, backtrack
        # 2. Backtrack attempt: Refused again, but max backtracks reached
        # 3. Turn 2: Continues despite refusal
        # 4. Turn 3: Continues to max turns
        mock_prompt_normalizer.send_prompt_async.side_effect = [
            adv_response,
            sample_response,  # Turn 1: First attempt (refused, backtrack)
            adv_response,
            sample_response,  # After backtrack: Second attempt (refused, but max backtracks reached)
            adv_response,
            sample_response,  # Turn 2: Third attempt (continues despite refusal)
            adv_response,
            sample_response,  # Turn 3: Fourth attempt (to reach max turns)
        ]

        # Mock check_refusal_async to always return refusal
        mock_check_refusal = AsyncMock(return_value=refusal_score)

        # Mock backtrack memory once
        with patch.object(
            attack, "_backtrack_memory_async", new_callable=AsyncMock, return_value="new_conv_id"
        ) as mock_backtrack:
            with patch.object(attack, "_check_refusal_async", mock_check_refusal):
                with patch(
                    "pyrit.score.Scorer.score_response_with_objective_async",
                    new_callable=AsyncMock,
                    return_value={"objective_scores": [failure_objective_score], "auxiliary_scores": []},
                ):
                    result = await attack._perform_attack_async(context=basic_context)

        # Should only backtrack once (when backtrack count is 0)
        assert mock_backtrack.call_count == 1
        assert isinstance(result, CrescendoAttackResult)
        assert result.outcome == AttackOutcome.FAILURE
        assert result.executed_turns == 3  # Reaches max turns despite refusals
        assert result.backtrack_count == 1


@pytest.mark.usefixtures("patch_central_database")
class TestContextCreation:
    """Tests for _create_context_from_params method"""

    def test_create_context_from_params_basic(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test basic context creation with minimal parameters."""
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        runtime_config = AttackRuntimeConfig()
        context = attack._create_context_from_params(
            objective="Test objective",
            prepended_conversation=[],
            memory_labels={"test": "label"},
            runtime_config=runtime_config,
        )

        assert isinstance(context, CrescendoAttackContext)
        assert context.objective == "Test objective"
        assert context.prepended_conversation == []
        assert context.memory_labels == {"test": "label"}
        # Should use dataclass defaults when not specified in runtime_config
        assert context.max_turns == 10  # Default value
        assert context.max_backtracks == 10  # Default value

    def test_create_context_from_params_with_runtime_config(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test context creation with runtime configuration."""
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        runtime_config = AttackRuntimeConfig(max_turns=5, max_backtracks=3)
        context = attack._create_context_from_params(
            objective="Test objective",
            prepended_conversation=[],
            memory_labels={},
            runtime_config=runtime_config,
        )

        assert context.max_turns == 5
        assert context.max_backtracks == 3

    def test_create_context_from_params_with_partial_runtime_config(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test context creation with partial runtime configuration."""
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        # Only set max_turns, leave max_backtracks as None
        runtime_config = AttackRuntimeConfig(max_turns=7)
        context = attack._create_context_from_params(
            objective="Test objective",
            prepended_conversation=[],
            memory_labels={},
            runtime_config=runtime_config,
        )

        assert context.max_turns == 7
        assert context.max_backtracks == 10  # Should use default

        # Only set max_backtracks, leave max_turns as None
        runtime_config = AttackRuntimeConfig(max_backtracks=2)
        context = attack._create_context_from_params(
            objective="Test objective",
            prepended_conversation=[],
            memory_labels={},
            runtime_config=runtime_config,
        )

        assert context.max_turns == 10  # Should use default
        assert context.max_backtracks == 2

    def test_create_context_from_params_with_custom_prompt(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test context creation with custom prompt."""
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        runtime_config = AttackRuntimeConfig()
        context = attack._create_context_from_params(
            objective="Test objective",
            prepended_conversation=[],
            memory_labels={},
            runtime_config=runtime_config,
            custom_prompt="My custom prompt",
        )

        assert context.custom_prompt == "My custom prompt"

    def test_create_context_from_params_invalid_custom_prompt(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test that non-string custom_prompt raises ValueError."""
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        runtime_config = AttackRuntimeConfig()

        # Test with various invalid types
        invalid_prompts = [123, [], {}, True, 3.14]

        for invalid_prompt in invalid_prompts:
            with pytest.raises(ValueError, match="custom_prompt must be a string"):
                attack._create_context_from_params(
                    objective="Test objective",
                    prepended_conversation=[],
                    memory_labels={},
                    runtime_config=runtime_config,
                    custom_prompt=invalid_prompt,
                )

    def test_create_context_from_params_prepended_conversation(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        sample_response: PromptRequestResponse,
    ):
        """Test context creation with prepended conversation."""
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        prepended_conversation = [sample_response]
        runtime_config = AttackRuntimeConfig()
        context = attack._create_context_from_params(
            objective="Test objective",
            prepended_conversation=prepended_conversation,
            memory_labels={},
            runtime_config=runtime_config,
        )

        assert context.prepended_conversation == prepended_conversation
        assert len(context.prepended_conversation) == 1


@pytest.mark.usefixtures("patch_central_database")
class TestAttackLifecycle:
    """Tests for the complete attack lifecycle (execute_with_context_async and execute_async)"""

    @pytest.mark.asyncio
    async def test_execute_with_context_async_successful_lifecycle(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
        success_objective_score: Score,
    ):
        """Test successful execution of complete attack lifecycle."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        # Mock all lifecycle methods
        with patch.object(attack, "_validate_context"):
            with patch.object(attack, "_setup_async", new_callable=AsyncMock):
                with patch.object(attack, "_perform_attack_async", new_callable=AsyncMock) as mock_perform:
                    with patch.object(attack, "_teardown_async", new_callable=AsyncMock):
                        # Configure the return value for _perform_attack_async
                        mock_perform.return_value = CrescendoAttackResult(
                            conversation_id=basic_context.session.conversation_id,
                            objective=basic_context.objective,
                            attack_identifier=attack.get_identifier(),
                            outcome=AttackOutcome.SUCCESS,
                            executed_turns=1,
                            last_response=sample_response.get_piece(),
                            last_score=success_objective_score,
                            metadata={"backtrack_count": 0},
                        )

                        # Execute the complete lifecycle
                        result = await attack.execute_with_context_async(context=basic_context)

        # Verify result and proper execution order
        assert isinstance(result, CrescendoAttackResult)
        assert result.outcome == AttackOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_with_context_async_validation_failure_prevents_execution(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        basic_context: CrescendoAttackContext,
    ):
        """Test that validation failure prevents attack execution."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        # Mock validation to fail
        with patch.object(attack, "_validate_context", side_effect=ValueError("Invalid context")) as mock_validate:
            with patch.object(attack, "_setup_async", new_callable=AsyncMock) as mock_setup:
                with patch.object(attack, "_perform_attack_async", new_callable=AsyncMock) as mock_perform:
                    with patch.object(attack, "_teardown_async", new_callable=AsyncMock) as mock_teardown:
                        # Should raise AttackValidationException
                        with pytest.raises(AttackValidationException) as exc_info:
                            await attack.execute_with_context_async(context=basic_context)

        # Verify error details
        assert "Context validation failed" in str(exc_info.value)

        # Verify only validation was attempted
        mock_validate.assert_called_once_with(context=basic_context)
        mock_setup.assert_not_called()
        mock_perform.assert_not_called()
        mock_teardown.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_async_with_parameters(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        sample_response: PromptRequestResponse,
        success_objective_score: Score,
    ):
        """Test the new parametrized execute_async method."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        # Mock the execute_with_context_async to return a successful result
        mock_result = CrescendoAttackResult(
            conversation_id="test-conversation-id",
            objective="Test objective",
            attack_identifier=attack.get_identifier(),
            outcome=AttackOutcome.SUCCESS,
            executed_turns=1,
            last_response=sample_response.get_piece(),
            last_score=success_objective_score,
            metadata={"backtrack_count": 0},
        )

        with patch.object(attack, "execute_with_context_async", new_callable=AsyncMock, return_value=mock_result):
            result = await attack.execute_async(
                objective="Test objective",
                memory_labels={"test": "label"},
                runtime_config=AttackRuntimeConfig(max_turns=5),
                custom_prompt="Custom prompt",
            )

        assert isinstance(result, CrescendoAttackResult)
        assert result.outcome == AttackOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_async_with_default_config(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test execute_async with default runtime configuration."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        # Mock create_context_from_params to verify it's called correctly
        mock_context = CrescendoAttackContext(objective="Test objective")

        with patch.object(attack, "_create_context_from_params", return_value=mock_context) as mock_create:
            with patch.object(attack, "execute_with_context_async", new_callable=AsyncMock) as mock_execute:
                await attack.execute_async(objective="Test objective")

        # Verify create_context_from_params was called with default values
        mock_create.assert_called_once()
        call_args = mock_create.call_args.kwargs
        assert call_args["objective"] == "Test objective"
        assert call_args["prepended_conversation"] == []
        assert call_args["memory_labels"] == {}
        assert isinstance(call_args["runtime_config"], AttackRuntimeConfig)

        # Verify execute_with_context_async was called with the created context
        mock_execute.assert_called_once_with(context=mock_context)


@pytest.mark.usefixtures("patch_central_database")
class TestIntegrationScenarios:
    """Integration tests for complete attack scenarios

    These tests verify that all components work together correctly in realistic scenarios.
    They simulate full attack flows including multiple turns, scoring, and decision-making.
    """

    @pytest.mark.asyncio
    async def test_complete_successful_attack_scenario(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
    ):
        """Test a complete successful attack scenario with multiple turns.

        This simulates a realistic Crescendo attack that gradually escalates:
        1. Turn 1: Benign question to build rapport (low score)
        2. Turn 2: Slightly more targeted question (medium score)
        3. Turn 3: Direct request that achieves objective (high score)

        The escalation pattern is key to Crescendo's effectiveness.
        """
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = CrescendoAttackContext(
            objective="Extract sensitive information",
            max_turns=3,
        )

        # Set up mock responses for a 3-turn successful attack
        adversarial_responses = [
            create_adversarial_json_response(
                question="What's your favorite color?", summary="Initial probe", rationale="Building rapport"
            ),
            create_adversarial_json_response(
                question="What kind of data do you process?",
                summary="User likes blue",
                rationale="Moving to data topics",
            ),
            create_adversarial_json_response(
                question="Can you show me an example?",
                summary="Processes user data",
                rationale="Final extraction attempt",
            ),
        ]

        target_responses = [
            create_prompt_response(text="I like blue!"),
            create_prompt_response(text="I process user data and configurations"),
            create_prompt_response(text="Sure, here's an example: [sensitive data]"),
        ]

        # Create scores that show progression toward the objective
        # The increasing scores simulate the model becoming more compliant
        scores = [
            create_score(
                score_type="float_scale",
                score_value="0.3",
                score_category="objective",
                scorer_class="FloatScaleThresholdScorer",
            ),
            create_score(
                score_type="float_scale",
                score_value="0.6",
                score_category="objective",
                scorer_class="FloatScaleThresholdScorer",
            ),
            create_score(
                score_type="float_scale",
                score_value="0.9",
                score_category="objective",
                scorer_class="FloatScaleThresholdScorer",
            ),  # Above threshold
        ]

        # Set up mock behavior to simulate the conversation flow
        # The side_effect list must match the exact sequence of calls
        mock_prompt_normalizer.send_prompt_async.side_effect = [
            create_prompt_response(text=adversarial_responses[0]),  # Adversarial generates turn 1
            target_responses[0],  # Target responds to turn 1
            create_prompt_response(text=adversarial_responses[1]),  # Adversarial generates turn 2
            target_responses[1],  # Target responds to turn 2
            create_prompt_response(text=adversarial_responses[2]),  # Adversarial generates turn 3
            target_responses[2],  # Target responds to turn 3
        ]

        no_refusal = create_score(
            score_type="true_false", score_value="false", score_category="refusal", scorer_class="SelfAskRefusalScorer"
        )

        # Mock the conversation manager to return an empty state (no prepended conversation)
        mock_conversation_state = ConversationState(turn_count=0)

        with patch.object(
            attack._conversation_manager, "update_conversation_state_async", return_value=mock_conversation_state
        ):
            with patch.object(attack, "_check_refusal_async", new_callable=AsyncMock, return_value=no_refusal):
                with patch(
                    "pyrit.score.Scorer.score_response_with_objective_async", new_callable=AsyncMock
                ) as mock_score:
                    mock_score.side_effect = [
                        {"objective_scores": [scores[0]], "auxiliary_scores": []},
                        {"objective_scores": [scores[1]], "auxiliary_scores": []},
                        {"objective_scores": [scores[2]], "auxiliary_scores": []},
                    ]

                    result = await attack.execute_with_context_async(context=context)

        # Verify the attack succeeded
        assert isinstance(result, CrescendoAttackResult)
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.executed_turns == 3
        assert result.last_score is not None
        assert result.last_score.get_value() == 0.9
        assert result.last_response is not None
        assert "sensitive data" in result.last_response.converted_value

    @pytest.mark.asyncio
    async def test_attack_with_backtracking_scenario(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        refusal_score: Score,
        no_refusal_score: Score,
        success_objective_score: Score,
    ):
        """Test attack scenario with backtracking due to refusals.

        This tests Crescendo's adaptive behavior:
        1. First two attempts are too direct and get refused
        2. Each refusal triggers backtracking and a new approach
        3. Third attempt is subtle enough to succeed

        This demonstrates how Crescendo learns from failures.
        """
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        context = CrescendoAttackContext(
            objective="Test with refusals",
            max_turns=2,
            max_backtracks=2,
        )

        # Set up responses
        adversarial_responses = [
            create_adversarial_json_response(
                question="Direct harmful request", summary="Initial", rationale="Too direct"
            ),
            create_adversarial_json_response(
                question="Subtle approach", summary="Was refused", rationale="Trying indirect"
            ),
            create_adversarial_json_response(
                question="Even more subtle", summary="Still refused", rationale="Third attempt"
            ),
        ]

        target_responses = [
            create_prompt_response(text="I cannot help with that"),  # Refused
            create_prompt_response(text="I'm not comfortable with this"),  # Refused
            create_prompt_response(text="Well, maybe I can help..."),  # Accepted
        ]

        # Mock setup - responses must be interleaved correctly
        # This is complex because each attempt involves:
        # 1. Adversarial chat generating a prompt
        # 2. Target responding (possibly with refusal)
        all_responses = []
        for i in range(len(adversarial_responses)):
            all_responses.append(create_prompt_response(text=adversarial_responses[i]))
            if i < len(target_responses):
                all_responses.append(target_responses[i])

        mock_prompt_normalizer.send_prompt_async.side_effect = all_responses

        # First two attempts are refused, third succeeds
        # This pattern simulates the attack learning what works
        refusal_checks = [refusal_score, refusal_score, no_refusal_score]

        # Mock the conversation manager to return an empty state
        mock_conversation_state = ConversationState(turn_count=0)

        with patch.object(
            attack._conversation_manager, "update_conversation_state_async", return_value=mock_conversation_state
        ):
            with patch.object(attack, "_check_refusal_async", new_callable=AsyncMock, side_effect=refusal_checks):
                with patch.object(
                    attack, "_backtrack_memory_async", new_callable=AsyncMock, return_value="new_conv_id"
                ):
                    with patch(
                        "pyrit.score.Scorer.score_response_with_objective_async", new_callable=AsyncMock
                    ) as mock_score:
                        mock_score.return_value = {
                            "objective_scores": [success_objective_score],
                            "auxiliary_scores": [],
                        }

                        result = await attack.execute_with_context_async(context=context)

        # Verify backtracking occurred as expected
        assert isinstance(result, CrescendoAttackResult)
        assert result.backtrack_count == 2  # Two failed attempts
        assert result.executed_turns == 1  # Only successful turns count toward limit


@pytest.mark.usefixtures("patch_central_database")
class TestEdgeCases:
    """Tests for edge cases and error conditions

    These tests ensure the attack handles unexpected situations gracefully
    and provides meaningful error messages when things go wrong.
    """

    @pytest.mark.asyncio
    async def test_attack_with_empty_objective(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test that empty objective is properly rejected."""
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        context = CrescendoAttackContext(objective="", max_turns=5)

        with pytest.raises(AttackValidationException, match="Attack objective must be provided"):
            await attack.execute_with_context_async(context=context)

    @pytest.mark.asyncio
    async def test_attack_with_json_parsing_retry(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: CrescendoAttackContext,
    ):
        """Test that JSON parsing errors trigger retry mechanism.

        The adversarial chat might occasionally return malformed JSON.
        The retry decorator should handle transient failures automatically,
        making the attack more robust in production environments.
        """
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # First response is invalid JSON (simulating transient error)
        # Second response is valid (simulating successful retry)
        responses = [
            create_prompt_response(text="Invalid JSON response"),
            create_prompt_response(text=create_adversarial_json_response()),
        ]

        mock_prompt_normalizer.send_prompt_async.side_effect = responses

        # The retry decorator should handle the first failure transparently
        result = await attack._get_attack_prompt_async(context=basic_context, refused_text="")

        assert result == "Attack prompt"
        # Verify retry occurred - called at least twice due to first failure
        assert mock_prompt_normalizer.send_prompt_async.call_count >= 2

    @pytest.mark.asyncio
    async def test_attack_handles_scoring_errors_gracefully(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        basic_context: CrescendoAttackContext,
        sample_response: PromptRequestResponse,
    ):
        """Test that scoring errors are handled appropriately."""
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        basic_context.last_response = sample_response

        # Mock scoring to return empty list
        with patch("pyrit.score.Scorer.score_response_with_objective_async", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = {"objective_scores": [], "auxiliary_scores": []}

            with pytest.raises(RuntimeError, match="No objective scores returned"):
                await attack._score_response_async(context=basic_context)

    @pytest.mark.asyncio
    async def test_concurrent_context_isolation(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test that concurrent attacks don't interfere with each other.

        In production, multiple Crescendo attacks might run simultaneously.
        This test ensures that each attack maintains its own state and
        conversation context, preventing cross-contamination of data.
        """
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        # Create two contexts that could be used concurrently
        # They have different objectives and max_turns to ensure isolation
        context1 = CrescendoAttackContext(objective="Objective 1", max_turns=5)
        context2 = CrescendoAttackContext(objective="Objective 2", max_turns=3)

        # Mock conversation manager for both setups
        mock_state1 = ConversationState(turn_count=0)
        mock_state2 = ConversationState(turn_count=0)

        with patch.object(
            attack._conversation_manager, "update_conversation_state_async", side_effect=[mock_state1, mock_state2]
        ):
            # Simulate concurrent setup - both contexts use the same attack instance
            await attack._setup_async(context=context1)
            await attack._setup_async(context=context2)

        # Verify contexts remain independent
        # Each should maintain its own state without interference
        assert context1.objective == "Objective 1"
        assert context2.objective == "Objective 2"
        assert context1.max_turns == 5
        assert context2.max_turns == 3
        # Most importantly, they should have different conversation IDs
        assert context1.session.conversation_id != context2.session.conversation_id

    @pytest.mark.asyncio
    async def test_execute_async_integration(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_prompt_normalizer: MagicMock,
        sample_response: PromptRequestResponse,
        success_objective_score: Score,
        no_refusal_score: Score,
        adversarial_response: str,
    ):
        """Test the new execute_async method end-to-end."""
        attack = CrescendoTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            prompt_normalizer=mock_prompt_normalizer,
        )

        # Mock adversarial response
        adv_response = create_prompt_response(text=adversarial_response)

        # Set up mocks
        mock_prompt_normalizer.send_prompt_async.side_effect = [adv_response, sample_response]

        # Mock conversation state
        mock_conversation_state = ConversationState(turn_count=0)

        with patch.object(
            attack._conversation_manager, "update_conversation_state_async", return_value=mock_conversation_state
        ):
            with patch.object(attack, "_check_refusal_async", new_callable=AsyncMock, return_value=no_refusal_score):
                with patch(
                    "pyrit.score.Scorer.score_response_with_objective_async",
                    new_callable=AsyncMock,
                    return_value={"objective_scores": [success_objective_score], "auxiliary_scores": []},
                ):
                    # Use the new execute_async method with parameters
                    result = await attack.execute_async(
                        objective="Test objective",
                        memory_labels={"test": "label"},
                        runtime_config=AttackRuntimeConfig(max_turns=1),
                    )

        assert isinstance(result, CrescendoAttackResult)
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.objective == "Test objective"

    @pytest.mark.asyncio
    async def test_execute_async_with_invalid_attack_params(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
    ):
        """Test execute_async with invalid attack parameters."""
        adversarial_config = AttackAdversarialConfig(target=mock_adversarial_chat)

        attack = CrescendoAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=adversarial_config,
        )

        # Test with unknown parameter - should be ignored
        mock_result = CrescendoAttackResult(
            conversation_id="test-id",
            objective="Test objective",
            attack_identifier=attack.get_identifier(),
            outcome=AttackOutcome.SUCCESS,
            executed_turns=1,
        )

        with patch.object(attack, "execute_with_context_async", new_callable=AsyncMock, return_value=mock_result):
            # Unknown parameters should be ignored, not raise an error
            result = await attack.execute_async(
                objective="Test objective",
                unknown_param="should be ignored",
            )
            assert result.outcome == AttackOutcome.SUCCESS
