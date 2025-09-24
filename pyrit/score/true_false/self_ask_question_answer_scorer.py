# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pathlib
from typing import Optional

from pyrit.common.path import SCORER_CONFIG_PATH
from pyrit.models import (
    Score,
    UnvalidatedScore,
)
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import SelfAskTrueFalseScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import (
    OR_,
    TrueFalseScoreAggregator,
)


class SelfAskQuestionAnswerScorer(SelfAskTrueFalseScorer):
    """
    A class that represents a self-ask question answering scorer.

    Usually, QuestionAnswerScorer should be used, but this can be useful for when choices are not sent
    to objective target or you need more flexibility in determining if the questions were answered correctly.
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(
        supported_data_types=["text"],
        is_objective_required=True,
    )

    def __init__(
        self,
        *,
        chat_target: PromptChatTarget,
        true_false_question_path: Optional[pathlib.Path] = None,
        validator: Optional[ScorerPromptValidator] = None,
        score_aggregator: TrueFalseScoreAggregator = OR_,
    ) -> None:
        """
        Initializes the SelfAskQuestionAnswerScorer object.

        Args:
            chat_target (PromptChatTarget): The chat target to use for the scorer.
            true_false_question_path (pathlib.Path): The path to the true/false question file.
        """

        true_false_question_path = self._verify_and_resolve_path(
            true_false_question_path
            if true_false_question_path
            else SCORER_CONFIG_PATH / "true_false_question" / "question_answering.yaml"
        )
        super().__init__(
            chat_target=chat_target,
            true_false_question_path=true_false_question_path,
            validator=validator,
            score_aggregator=score_aggregator,
        )

    async def _score_piece_async(
        self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None
    ) -> list[Score]:
        """
        Score the request_reponse using the QuestionAnsweringEntry
        and return a single score object

        Args:
            request_response (PromptRequestPiece): The answer given by the target
            task (QuestionAnsweringEntry): The objective, which usually contains the question and the correct answer
        Returns:
            Score: A single Score object representing the result
        """

        prompt = (
            f"{objective}\n\nThe target response is as follows. "
            f"Evaluate if the response is correct:\n{request_piece.converted_value}"
        )

        unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=self._system_prompt,
            prompt_request_value=prompt,
            prompt_request_data_type="text",
            scored_prompt_id=request_piece.id,
            category=self._score_category,
            objective=objective,
            attack_identifier=request_piece.attack_identifier,
        )

        score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value, score_type="true_false")

        return [score]
