# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.attack import AttackAdversarialConfig, AttackScoringConfig
from pyrit.executor.attack.multi_turn.crescendo_score_conversation import (
    ConversationScoringAttack,
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


def create_mock_chat_target(*, name: str = "MockChatTarget") -> MagicMock:
    """Create a mock chat target with common setup."""
    target = MagicMock(spec=PromptChatTarget)
    target.send_prompt_async = AsyncMock()
    target.set_system_prompt = MagicMock()
    target.get_identifier.return_value = {"__type__": name, "__module__": "test_module"}
    return target


def create_mock_scorer(*, scorer_type: str, class_name: str) -> MagicMock:
    """Create a mock scorer with common setup."""
    scorer = MagicMock()
    scorer.scorer_type = scorer_type
    scorer.score_text_async = AsyncMock()
    scorer.get_identifier.return_value = {"__type__": class_name, "__module__": "test_module"}
    return scorer


def create_score(
    *,
    score_type: ScoreType,
    score_value: str,
    score_category: str = "objective",
    scorer_class: str = "TestScorer",
    score_rationale: str = "Test rationale",
    score_value_description: str = "Test description",
) -> Score:
    """Create a score with common defaults."""
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


def create_prompt_response(
    *, text: str, role: ChatMessageRole = "assistant", conversation_id: str = None, sequence: int = 1
) -> PromptRequestResponse:
    """Create a prompt response with common defaults."""
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role=role,
                original_value=text,
                original_value_data_type="text",
                converted_value=text,
                converted_value_data_type="text",
                conversation_id=conversation_id,
                sequence=sequence,
                timestamp=datetime.now(),
            )
        ]
    )


def create_conversation_history(conversation_id: str) -> list[PromptRequestResponse]:
    """Create a mock conversation history with multiple exchanges."""
    return [
        create_prompt_response(
            text="Hello, how can I help you today?", role="assistant", conversation_id=conversation_id, sequence=1
        ),
        create_prompt_response(
            text="I need help with something harmful.", role="user", conversation_id=conversation_id, sequence=2
        ),
        create_prompt_response(
            text="I can only give you part 1 of that", role="assistant", conversation_id=conversation_id, sequence=3
        ),
        create_prompt_response(text="What's part 2?.", role="user", conversation_id=conversation_id, sequence=4),
        create_prompt_response(
            text="Sure, here's how to do part 2 of that harmful thing...",
            role="assistant",
            conversation_id=conversation_id,
            sequence=5,
        ),
    ]


class ConversationScoringTestHelper:
    """Helper class for creating ConversationScoringAttack instances."""

    @staticmethod
    def create_attack(
        *,
        objective_target: MagicMock,
        adversarial_chat: MagicMock,
        objective_scorer: MagicMock = None,
        refusal_scorer: MagicMock = None,
        prompt_normalizer: MagicMock = None,
        successful_objective_threshold: float = 0.7,
        **kwargs,
    ) -> ConversationScoringAttack:
        """Create a ConversationScoringAttack instance with flexible configuration."""
        adversarial_config = AttackAdversarialConfig(target=adversarial_chat)

        scoring_config = None
        if objective_scorer or refusal_scorer:
            scoring_config = AttackScoringConfig(
                objective_scorer=objective_scorer,
                refusal_scorer=refusal_scorer,
                successful_objective_threshold=successful_objective_threshold,
                **kwargs,
            )

        attack = ConversationScoringAttack(
            objective_target=objective_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
            prompt_normalizer=prompt_normalizer,
        )

        ConversationScoringTestHelper.mock_memory_for_attack(attack)
        return attack

    @staticmethod
    def mock_memory_for_attack(attack: ConversationScoringAttack) -> MagicMock:
        """Mock the memory interface for the attack."""
        mock_memory = MagicMock()
        attack._memory = mock_memory
        return mock_memory


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
def sample_response() -> PromptRequestResponse:
    return create_prompt_response(text="Test response")


@pytest.fixture
def conversation_id() -> str:
    return str(uuid.uuid4())


@pytest.mark.usefixtures("patch_central_database")
class TestConversationScoringAttackInitialization:
    """Tests for ConversationScoringAttack initialization and configuration."""

    def test_init_inherits_from_crescendo(self, mock_objective_target: MagicMock, mock_adversarial_chat: MagicMock):
        """Test that ConversationScoringAttack properly inherits from CrescendoAttack."""
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )

        assert attack._objective_target == mock_objective_target
        assert attack._adversarial_chat == mock_adversarial_chat
        # Should inherit default scorers from parent CrescendoAttack
        assert attack._objective_scorer is not None
        assert attack._refusal_scorer is not None

    def test_init_with_custom_scoring_configuration(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        mock_refusal_scorer: MagicMock,
    ):
        """Test initialization with custom scorers."""
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
            refusal_scorer=mock_refusal_scorer,
        )

        assert attack._objective_scorer == mock_objective_scorer
        assert attack._refusal_scorer == mock_refusal_scorer

    def test_init_with_custom_threshold(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
    ):
        """Test initialization with custom successful_objective_threshold."""
        custom_threshold = 0.85
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
            successful_objective_threshold=custom_threshold,
        )

        assert attack._successful_objective_threshold == custom_threshold


@pytest.mark.usefixtures("patch_central_database")
class TestConversationScoringObjectiveEvaluation:
    """Tests for the conversation-level objective scoring functionality."""

    @pytest.mark.asyncio
    async def test_is_objective_achieved_no_scorer(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        sample_response: PromptRequestResponse,
    ):
        """Test that is_objective_achieved returns False when no objective scorer is available."""
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
        )
        # Remove the default scorer to test this case
        attack._objective_scorer = None

        result = await attack.is_objective_achieved(sample_response)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_objective_achieved_builds_conversation_text(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        conversation_id: str,
    ):
        """Test that the method properly builds conversation text from memory."""
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        # Create a conversation history
        conversation_history = create_conversation_history(conversation_id)
        attack._memory.get_conversation.return_value = conversation_history  # type: ignore

        # Mock the scorer to return a score
        success_score = create_score(score_type="float_scale", score_value="0.8")
        mock_objective_scorer.score_text_async.return_value = [success_score]

        # Create a sample response from the conversation
        response = create_prompt_response(text="Test response", conversation_id=conversation_id)

        result = await attack.is_objective_achieved(response)

        # Verify get_conversation was called with correct conversation_id
        attack._memory.get_conversation.assert_called_once_with(conversation_id=conversation_id)  # type: ignore

        # Verify the scorer was called with the complete conversation text
        mock_objective_scorer.score_text_async.assert_called_once()
        args, kwargs = mock_objective_scorer.score_text_async.call_args

        # Check that the conversation text includes all parts
        conversation_text = kwargs["text"]
        assert "Assistant: Hello, how can I help you today?" in conversation_text
        assert "User: I need help with something harmful." in conversation_text
        assert "Assistant: I can only give you part 1 of that" in conversation_text
        assert "User: What's part 2?." in conversation_text
        assert "Assistant: Sure, here's how to do part 2 of that harmful thing..." in conversation_text
        assert kwargs["task"] == "evaluate entire conversation"

        assert result is True  # Score 0.8 >= default threshold 0.7

    @pytest.mark.asyncio
    async def test_is_objective_achieved_true_false_score_true(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        conversation_id: str,
    ):
        """Test true_false scorer returning True (objective achieved)."""
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        conversation_history = create_conversation_history(conversation_id)
        attack._memory.get_conversation.return_value = conversation_history  # type: ignore

        # Mock true_false score with value "true"
        true_score = create_score(score_type="true_false", score_value="true")
        mock_objective_scorer.score_text_async.return_value = [true_score]

        response = create_prompt_response(text="Test response", conversation_id=conversation_id)
        result = await attack.is_objective_achieved(response)

        assert result is True

    @pytest.mark.asyncio
    async def test_is_objective_achieved_true_false_score_false(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        conversation_id: str,
    ):
        """Test true_false scorer returning False (objective not achieved)."""
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        conversation_history = create_conversation_history(conversation_id)
        attack._memory.get_conversation.return_value = conversation_history  # type: ignore

        # Mock true_false score with value "false"
        false_score = create_score(score_type="true_false", score_value="false")
        mock_objective_scorer.score_text_async.return_value = [false_score]

        response = create_prompt_response(text="Test response", conversation_id=conversation_id)
        result = await attack.is_objective_achieved(response)

        assert result is False

    @pytest.mark.asyncio
    async def test_is_objective_achieved_float_score_above_threshold(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        conversation_id: str,
    ):
        """Test float_scale scorer with score above threshold (objective achieved)."""
        custom_threshold = 0.6
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
            successful_objective_threshold=custom_threshold,
        )

        conversation_history = create_conversation_history(conversation_id)
        attack._memory.get_conversation.return_value = conversation_history  # type: ignore

        # Mock float_scale score above threshold
        high_score = create_score(score_type="float_scale", score_value="0.8")
        mock_objective_scorer.score_text_async.return_value = [high_score]

        response = create_prompt_response(text="Test response", conversation_id=conversation_id)
        result = await attack.is_objective_achieved(response)

        assert result is True

    @pytest.mark.asyncio
    async def test_is_objective_achieved_float_score_below_threshold(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        conversation_id: str,
    ):
        """Test float_scale scorer with score below threshold (objective not achieved)."""
        custom_threshold = 0.8
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
            successful_objective_threshold=custom_threshold,
        )

        conversation_history = create_conversation_history(conversation_id)
        attack._memory.get_conversation.return_value = conversation_history  # type: ignore

        # Mock float_scale score below threshold
        low_score = create_score(score_type="float_scale", score_value="0.5")
        mock_objective_scorer.score_text_async.return_value = [low_score]

        response = create_prompt_response(text="Test response", conversation_id=conversation_id)
        result = await attack.is_objective_achieved(response)

        assert result is False

    @pytest.mark.asyncio
    async def test_is_objective_achieved_float_score_at_threshold(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        conversation_id: str,
    ):
        """Test float_scale scorer with score exactly at threshold (objective achieved)."""
        custom_threshold = 0.7
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
            successful_objective_threshold=custom_threshold,
        )

        conversation_history = create_conversation_history(conversation_id)
        attack._memory.get_conversation.return_value = conversation_history  # type: ignore

        # Mock float_scale score exactly at threshold
        threshold_score = create_score(score_type="float_scale", score_value="0.7")
        mock_objective_scorer.score_text_async.return_value = [threshold_score]

        response = create_prompt_response(text="Test response", conversation_id=conversation_id)
        result = await attack.is_objective_achieved(response)

        assert result is True

    @pytest.mark.asyncio
    async def test_is_objective_achieved_unknown_score_type_fallback(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        conversation_id: str,
    ):
        """Test the fallback behavior for unrecognized score types."""
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        conversation_history = create_conversation_history(conversation_id)
        attack._memory.get_conversation.return_value = conversation_history  # type: ignore

        # Create a valid score and then mock its score_type to simulate unknown type
        mock_score = MagicMock()
        mock_score.score_type = "unknown_custom_type"  # This simulates an unknown score type
        mock_score.get_value.return_value = "some_value"
        mock_objective_scorer.score_text_async.return_value = [mock_score]

        response = create_prompt_response(text="Test response", conversation_id=conversation_id)
        result = await attack.is_objective_achieved(response)

        assert result is False  # Should fall back to False for unknown score types

    @pytest.mark.asyncio
    async def test_is_objective_achieved_no_scores_returned(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        conversation_id: str,
    ):
        """Test that no scores returned by scorer results in False."""
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        conversation_history = create_conversation_history(conversation_id)
        attack._memory.get_conversation.return_value = conversation_history  # type: ignore

        # Mock scorer returning empty list
        mock_objective_scorer.score_text_async.return_value = []

        response = create_prompt_response(text="Test response", conversation_id=conversation_id)
        result = await attack.is_objective_achieved(response)

        assert result is False

    @pytest.mark.asyncio
    async def test_is_objective_achieved_empty_conversation(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        conversation_id: str,
    ):
        """Test behavior with empty conversation history."""
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        # Mock empty conversation
        attack._memory.get_conversation.return_value = []  # type: ignore

        # Mock scorer to return a score
        success_score = create_score(score_type="float_scale", score_value="0.8")
        mock_objective_scorer.score_text_async.return_value = [success_score]

        response = create_prompt_response(text="Test response", conversation_id=conversation_id)
        result = await attack.is_objective_achieved(response)

        # Should still call scorer with empty conversation text
        mock_objective_scorer.score_text_async.assert_called_once()
        args, kwargs = mock_objective_scorer.score_text_async.call_args
        assert kwargs["text"] == ""  # Empty conversation should result in empty text

        assert result is True  # Score itself is still valid

    @pytest.mark.asyncio
    async def test_conversation_text_format(
        self,
        mock_objective_target: MagicMock,
        mock_adversarial_chat: MagicMock,
        mock_objective_scorer: MagicMock,
        conversation_id: str,
    ):
        """Test that conversation text is formatted correctly with role prefixes."""
        attack = ConversationScoringTestHelper.create_attack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )

        # Create conversation with specific content
        conversation_history = [
            create_prompt_response(
                text="How can I help?", role="assistant", conversation_id=conversation_id, sequence=1
            ),
            create_prompt_response(text="I need assistance", role="user", conversation_id=conversation_id, sequence=2),
            create_prompt_response(
                text="Sure, I'll help", role="assistant", conversation_id=conversation_id, sequence=3
            ),
        ]
        attack._memory.get_conversation.return_value = conversation_history  # type: ignore

        success_score = create_score(score_type="float_scale", score_value="0.8")
        mock_objective_scorer.score_text_async.return_value = [success_score]

        response = create_prompt_response(text="Test response", conversation_id=conversation_id)
        await attack.is_objective_achieved(response)

        # Check the exact format of the conversation text
        args, kwargs = mock_objective_scorer.score_text_async.call_args
        conversation_text = kwargs["text"]

        expected_text = "Assistant: How can I help?\n" "User: I need assistance\n" "Assistant: Sure, I'll help\n"
        assert conversation_text == expected_text
