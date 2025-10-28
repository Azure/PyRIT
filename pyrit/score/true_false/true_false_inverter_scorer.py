# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import Optional

from pyrit.models import ChatMessageRole, Message, MessagePiece, Score
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer


class TrueFalseInverterScorer(TrueFalseScorer):
    """A scorer that inverts a true false score."""

    def __init__(self, *, scorer: TrueFalseScorer, validator: Optional[ScorerPromptValidator] = None) -> None:
        """Initialize the TrueFalseInverterScorer.

        Args:
            scorer (TrueFalseScorer): The underlying true/false scorer whose results will be inverted.
            validator (Optional[ScorerPromptValidator]): Custom validator. Defaults to None.
                Note: This parameter is present for signature compatibility but is not used.
        """

        super().__init__(validator=ScorerPromptValidator())

        if not isinstance(scorer, TrueFalseScorer):
            raise ValueError("The scorer must be a true false scorer")
        self._scorer = scorer

    async def _score_async(
        self,
        message: Message,
        *,
        objective: Optional[str] = None,
        role_filter: Optional[ChatMessageRole] = None,
    ) -> list[Score]:
        """Scores the piece using the underlying true-false scorer and returns the inverted score.

        Args:
            message (Message): The message to score.
            objective (Optional[str]): The objective to evaluate against (the original attacker model's objective).
                Defaults to None.
            role_filter (Optional[ChatMessageRole]): Optional filter for message roles. Defaults to None.

        Returns:
            list[Score]: A list containing a single Score object with the inverted true/false value.
        """
        scores = await self._scorer.score_async(
            message,
            objective=objective,
            role_filter=role_filter,
        )

        # TrueFalseScorers only have a single score
        inv_score = scores[0]

        inv_score.score_value = str(True) if not inv_score.get_value() else str(False)
        inv_score.score_value_description = "Inverted score: " + str(inv_score.score_value_description)

        scorer_type = self._scorer.get_identifier().get("__type__", "Unknown")
        inv_score.score_rationale = (
            f"Inverted score from {scorer_type} result: {inv_score.score_value}\n" f"{inv_score.score_rationale}"
        )

        inv_score.id = uuid.uuid4()

        inv_score.scorer_class_identifier = self.get_identifier()

        return [inv_score]

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """True False Inverter scorers do not support piecewise scoring.

        Args:
            message_piece (MessagePiece): Unused.
            objective (Optional[str]): Unused.

        Raises:
            NotImplementedError: Always, since composite scoring operates at the response level.
        """
        raise NotImplementedError("TrueFalseCompositeScorer does not support piecewise scoring.")

    def _get_sub_identifier(self):
        """
        Returns the identifier of the underlying true/false scorer.

        Returns:
            dict: The identifier dictionary of the wrapped scorer.
        """
        return self._scorer.get_identifier()
