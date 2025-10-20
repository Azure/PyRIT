# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
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

    async def _score_piece_async(self, request_piece: PromptRequestPiece, *, objective: str = None) -> list[Score]:
        """
        Required abstract method - scores the entire conversation history.

        Args:
            request_piece: A piece from the conversation to be scored.
                The conversation ID is used to retrieve the full conversation from memory.
            objective: Optional objective to evaluate against.

        Returns:
            List of Score objects from the underlying scorer
        """
        # Retrieve the full conversation from memory using the conversation_id
        conversation = self._memory.get_conversation(conversation_id=request_piece.conversation_id)

        if not conversation:
            raise ValueError(f"Conversation with ID {request_piece.conversation_id} not found in memory.")

        # Build the full conversation text
        conversation_text = ""

        # Sort by timestamp to get chronological order
        sorted_conversation = sorted(conversation, key=lambda x: x.request_pieces[0].timestamp)

        for response in sorted_conversation:
            for piece in response.request_pieces:
                if piece.role == "user":
                    conversation_text += f"User: {piece.converted_value}\n"
                elif piece.role == "assistant":
                    conversation_text += f"Assistant: {piece.converted_value}\n"

        # Create a modified copy of the request piece with the full conversation text
        modified_piece = PromptRequestPiece(
            role=request_piece.role,
            original_value=conversation_text,
            converted_value=conversation_text,
            id=request_piece.id,  # Keep the original ID so memory lookups work
            conversation_id=request_piece.conversation_id,
            sequence=request_piece.sequence,
            labels=request_piece.labels,
            prompt_metadata=request_piece.prompt_metadata,
            converter_identifiers=request_piece.converter_identifiers,
            prompt_target_identifier=request_piece.prompt_target_identifier,
            attack_identifier=request_piece.attack_identifier,
            scorer_identifier=request_piece.scorer_identifier,
            original_value_data_type=request_piece.original_value_data_type,
            converted_value_data_type=request_piece.converted_value_data_type,
            response_error=request_piece.response_error,
            originator=request_piece.originator,
            original_prompt_id=request_piece.original_prompt_id,
            timestamp=request_piece.timestamp,
        )

        scores = await self._scorer._score_piece_async(request_piece=modified_piece, objective=objective)

        return scores

    async def score_text_async(self, text: str, *, objective: str = None) -> list[Score]:
        """
        For direct text scoring, just delegate to the underlying scorer.
        This is called when scoring outside of a conversation context.
        """
        return await self._scorer.score_text_async(text=text, objective=objective)

    def validate(self, request_response: PromptRequestResponse):
        """Validate using the underlying scorer's validation"""
        return self._scorer.validate(request_response)

    def validate_return_scores(self, scores: list[Score]):
        """
        Validate the scores returned by the scorer.
        Delegates to the underlying scorer's validation logic.
        """
        return self._scorer.validate_return_scores(scores)
