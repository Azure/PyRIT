# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, cast
from uuid import UUID

from pyrit.models import Message, MessagePiece, Score
from pyrit.models.literals import PromptResponseError
from pyrit.models.message_piece import Originator
from pyrit.score import Scorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator


class ConversationScorer(Scorer):
    """
    Custom scorer that evaluates the entire conversation history rather than just
    the latest response. This is useful for evaluating multi-turn conversations
    where context matters (e.g., psychosocial harms that emerge over time).

    This scorer wraps another scorer (like SelfAskLikertScorer or SelfAskGeneralFloatScaleScorer) and
    feeds it the full conversation history before scoring.

    Similar to LookBackScorer, but does not specifically look for behavior changes.
    It combines the entire conversation and makes the text into one string
    in order to score the conversation as a whole.
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(
        supported_data_types=["text"],
        max_pieces_in_response=1,
        enforce_all_pieces_valid=True,
    )

    def __init__(
        self,
        *,
        scorer: Scorer,
        validator: Optional[ScorerPromptValidator] = None,
    ):
        """
        Args:
            scorer (Scorer): The underlying scorer to use for evaluation
            validator (Optional[ScorerPromptValidator]): Optional validator. Defaults to base ScorerPromptValidator.
        """
        super().__init__(validator=validator or self._default_validator)
        self._scorer = scorer

    async def _score_async(self, message: Message, *, objective: Optional[str] = None) -> list[Score]:
        """
        Scores the entire conversation history by concatenating all messages and passing to the wrapped scorer.

        Args:
            message (Message): A message from the conversation to be scored.
                The conversation ID from the first message piece is used to retrieve the full conversation from memory.
            objective (Optional[str]): Optional objective to evaluate against.

        Returns:
            list[Score]: List of Score objects from the underlying scorer
        """
        if not message.message_pieces:
            return []

        # Get conversation ID from the first message piece
        conversation_id = message.message_pieces[0].conversation_id

        # Retrieve the full conversation from memory using the conversation_id
        conversation = self._memory.get_conversation(conversation_id=conversation_id)

        if not conversation:
            raise ValueError(f"Conversation with ID {conversation_id} not found in memory.")

        # Build the full conversation text
        conversation_text = ""

        # Goes through each message in the conversation and appends user and assistant messages
        for conv_message in conversation:
            for piece in conv_message.message_pieces:
                if piece.role == "user":
                    conversation_text += f"User: {piece.converted_value}\n"
                elif piece.role == "assistant":
                    conversation_text += f"Assistant: {piece.converted_value}\n"

        # Create a new message with the concatenated conversation text
        # Preserve the original message piece metadata
        original_piece = message.message_pieces[0]
        conversation_message = Message(
            message_pieces=[
                MessagePiece(
                    role=original_piece.role,
                    original_value=conversation_text,
                    converted_value=conversation_text,
                    id=original_piece.id,
                    conversation_id=original_piece.conversation_id,
                    labels=original_piece.labels,
                    prompt_target_identifier=original_piece.prompt_target_identifier,
                    attack_identifier=original_piece.attack_identifier,
                    original_value_data_type=original_piece.original_value_data_type,
                    converted_value_data_type=original_piece.converted_value_data_type,
                    response_error=cast(PromptResponseError, original_piece.response_error),
                    originator=cast(Originator, original_piece.originator),
                    original_prompt_id=(
                        cast(UUID, original_piece.original_prompt_id)
                        if isinstance(original_piece.original_prompt_id, str)
                        else original_piece.original_prompt_id
                    ),
                    timestamp=original_piece.timestamp,
                )
            ]
        )

        # Score using the underlying scorer's _score_async method (not score_async)
        # This prevents double-insertion into the database since the parent's score_async
        # will handle adding scores to memory
        scores = await self._scorer._score_async(message=conversation_message, objective=objective)

        return scores

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """
        Required abstract method - not used for ConversationScorer as we override _score_async.
        """
        raise NotImplementedError("ConversationScorer uses _score_async, not _score_piece_async")

    def validate_return_scores(self, scores: list[Score]):
        """
        Validates the scores returned by the wrapped scorer.
        Delegates to the underlying scorer's validation.
        """
        self._scorer.validate_return_scores(scores=scores)
