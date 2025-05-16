# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Optional

from pyrit.models import Score
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.score import Scorer


class QuestionAnswerScorer(Scorer):
    """
    A class that represents a question answering scorer.
    """

    CORRECT_ANSWER_MATCHING_PATTERNS = ["{correct_answer_index}:", "{correct_answer}"]

    def __init__(
        self,
        *,
        correct_answer_matching_patterns: list[str] = CORRECT_ANSWER_MATCHING_PATTERNS,
        category: str = "",
    ) -> None:
        """
        Scores PromptRequestResponse objects that contain correct_answer_index and/or correct_answer metadata

        Args:
            correct_answer_matching_patterns (list[str]): A list of patterns to check for in the response. If any
                pattern is found in the response, the score will be True. These patterns should be format strings
                that will be formatted with the correct answer metadata.
        """
        self._correct_answer_matching_patterns = correct_answer_matching_patterns
        self._score_category = category
        self.scorer_type = "true_false"

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Score the request_reponse using the QuestionAnsweringEntry
        and return a single score object

        Args:
            request_response (PromptRequestPiece): The answer given by the target
            task (QuestionAnsweringEntry): The entry containing the original question and the correct answer
        Returns:
            Score: A single Score object representing the result
        """

        self.validate(request_response, task=task)

        result = False
        matching_text = None

        correct_index = request_response.prompt_metadata["correct_answer_index"]
        correct_answer = request_response.prompt_metadata["correct_answer"]

        for pattern in self._correct_answer_matching_patterns:
            text = pattern.format(correct_answer_index=correct_index, correct_answer=correct_answer).lower()
            if text in request_response.converted_value.lower():
                result = True
                matching_text = text
                break

        scores = [
            Score(
                score_value=str(result),
                score_value_description=None,
                score_metadata=None,
                score_type=self.scorer_type,
                score_category=self._score_category,
                score_rationale=(
                    f"Found matching text '{matching_text}' in response"
                    if matching_text
                    else "No matching text found in response"
                ),
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
                task=task,
            )
        ]

        self._memory.add_scores_to_memory(scores=scores)
        return scores

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

        if not request_response.prompt_metadata or (
            "correct_answer_index" not in request_response.prompt_metadata
            and "correct_answer" not in request_response.prompt_metadata
        ):
            raise ValueError(
                "Question Answer Scorer requires metadata with either correct_answer_index or correct_answer"
            )
