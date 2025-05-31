# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import (
    Score,
    UnvalidatedScore,
)
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import SelfAskTrueFalseScorer


class SelfAskQuestionAnswerScorer(SelfAskTrueFalseScorer):
    """
    A class that represents a self-ask question answering scorer.

    Usually, QuestionAnswerScorer should be used, but this can be useful for when choices are not sent
    to objective target or you need more flexibility in determining if the questions were answered correctly.
    """

    def __init__(
        self,
        *,
        chat_target: PromptChatTarget,
        true_false_question_path: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Initializes the SelfAskQuestionAnswerScorer object.

        Args:
            chat_target (PromptChatTarget): The chat target to use for the scorer.
            true_false_question_path (pathlib.Path): The path to the true/false question file.
        """

        if not true_false_question_path:
            true_false_question_path = pathlib.Path(
                DATASETS_PATH, "score", "true_false_question", "question_answering.yaml"
            )

        super().__init__(
            chat_target=chat_target,
            true_false_question_path=true_false_question_path,
        )

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Score the request_reponse using the QuestionAnsweringEntry
        and return a single score object

        Args:
            request_response (PromptRequestPiece): The answer given by the target
            task (QuestionAnsweringEntry): The objective, which usually contains the question and the correct answer
        Returns:
            Score: A single Score object representing the result
        """

        self.validate(request_response, task=task)

        prompt = (
            f"{task}\n\nThe target response is as follows. "
            f"Evaluate if the response is correct:\n{request_response.converted_value}"
        )

        unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=self._system_prompt,
            prompt_request_value=prompt,
            prompt_request_data_type="text",
            scored_prompt_id=request_response.id,
            category=self._score_category,
            task=task,
            orchestrator_identifier=request_response.orchestrator_identifier,
        )

        score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value)

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        """
        Validates the request_response piece to score. Because some scorers may require
        specific PromptRequestPiece types or values.

        Args:
            request_response (PromptRequestPiece): The request response to be validated.
            task (str): The task based on which the text should be scored (the original attacker model's objective).
        """
        if request_response.converted_value_data_type != "text":
            raise ValueError("Question Answer Scorer only supports text data type")

        if not task:
            raise ValueError("Task must be provided")
