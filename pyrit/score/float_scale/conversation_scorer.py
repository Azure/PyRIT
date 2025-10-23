# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, cast
from uuid import UUID

from pyrit.models import Message, MessagePiece, Score
from pyrit.models.literals import PromptResponseError
from pyrit.models.message_piece import Originator
from pyrit.score import FloatScaleScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator


class ConversationHistoryScorer(FloatScaleScorer):
    """
    Custom scorer that evaluates the entire conversation history rather than just
    the latest response. This is useful for evaluating multi-turn conversations
    where context matters (e.g., psychosocial harms that emerge over time).

    This scorer wraps another scorer (like SelfAskGeneralFloatScaleScorer) and
    feeds it the full conversation history before scoring.

    Similar to LookBackScorer, but does not specifically look for behavior changes.
    """

    def __init__(self, scorer: FloatScaleScorer):
        """
        Args:
            scorer: The underlying scorer to use for evaluation (e.g., crisis_scorer)
        """
        # Initialize base Scorer with a default validator
        super().__init__(validator=ScorerPromptValidator())

        self._scorer = scorer

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """
        Required abstract method - scores the entire conversation history.

        Args:
            message_piece: A piece from the conversation to be scored.
                The conversation ID is used to retrieve the full conversation from memory.
            objective: Optional objective to evaluate against.

        Returns:
            List of Score objects from the underlying scorer
        """
        # Retrieve the full conversation from memory using the conversation_id
        conversation = self._memory.get_conversation(conversation_id=message_piece.conversation_id)

        if not conversation:
            raise ValueError(f"Conversation with ID {message_piece.conversation_id} not found in memory.")

        # Build the full conversation text
        conversation_text = ""

        # Sort by timestamp to get chronological order
        sorted_conversation = sorted(conversation, key=lambda x: x.message_pieces[0].timestamp)

        for message in sorted_conversation:
            for piece in message.message_pieces:
                if piece.role == "user":
                    conversation_text += f"User: {piece.converted_value}\n"
                elif piece.role == "assistant":
                    conversation_text += f"Assistant: {piece.converted_value}\n"

        # Create a modified copy of the message piece with the full conversation text
        # This preserves the original message piece metadata while replacing the content
        modified_piece = MessagePiece(
            role=message_piece.role,
            original_value=conversation_text,
            converted_value=conversation_text,
            id=message_piece.id,  # Preserve the original ID so memory lookups work
            conversation_id=message_piece.conversation_id,
            labels=message_piece.labels,
            prompt_target_identifier=message_piece.prompt_target_identifier,
            attack_identifier=message_piece.attack_identifier,
            original_value_data_type=message_piece.original_value_data_type,
            converted_value_data_type=message_piece.converted_value_data_type,
            # Use the original message piece's attributes for proper database tracking
            response_error=cast(PromptResponseError, message_piece.response_error),
            originator=cast(Originator, message_piece.originator),
            original_prompt_id=(
                cast(UUID, message_piece.original_prompt_id)
                if isinstance(message_piece.original_prompt_id, str)
                else message_piece.original_prompt_id
            ),
            timestamp=message_piece.timestamp,
        )

        # Score using the underlying scorer with the modified piece
        scores = await self._scorer._score_piece_async(message_piece=modified_piece, objective=objective)

        return scores

    async def score_text_async(self, text: str, *, objective: Optional[str] = None) -> list[Score]:
        """
        For direct text scoring, just delegate to the underlying scorer.
        This is called when scoring outside of a conversation context.
        """
        return await self._scorer.score_text_async(text=text, objective=objective)

    def validate(self, message: Message):
        """Validate using the underlying scorer's validation"""
        # Use the validator directly since scorers don't have a validate method
        return self._validator.validate(message, objective=None)

    def validate_return_scores(self, scores: list[Score]):
        """
        Validate the scores returned by the scorer.
        Delegates to the underlying scorer's validation logic.
        """
        return self._scorer.validate_return_scores(scores)
