# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from typing import Optional, Sequence

from pyrit.common.batch_helper import batch_task_async
from pyrit.models import QuestionAnsweringEntry, Score
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.score.scorer import Scorer


class QuestionAnswerScorer(Scorer):
    """
    A class that represents a question answering scorer.
    """

    def __init__(self, category: str = None) -> None:
        """
        Initializes the QuestionAnswerScorer object.
        """
        self._score_category = category
        self.scorer_type = "true_false"

    async def score_async(  # type: ignore[override]
        self, request_response: PromptRequestPiece, task: QuestionAnsweringEntry
    ) -> list[Score]:
        """
        Score the request_reponse using the QuestionAnsweringEntry
        and return a since Score object 30

        Args:
            request_response (PromptRequestPiece): The answer given by the target
            task (QuestionAnsweringEntry): The entry containing the original prompt and the correct answer
        Returns:
            Score: A single Score object representing the result
        """
        answer = request_response.converted_value
        answer_correct = False
        try:
            # This is the case where the model response is an integer, which is the index of the answer.
            answer = task.choices[int(answer)].text
        except ValueError:
            # If the model response is not an integer, then the model might have returned the answer as a string
            pass

        correct_answer = task.choices[int(task.correct_answer)].text
        metadata_json = {"question": str(task.question), "correct_answer": correct_answer, "scored_answer": answer}
        metadata = json.dumps(metadata_json)

        answer_correct = correct_answer in answer

        score = [
            Score(
                score_value=str(answer_correct),
                score_type=self.scorer_type,
                score_value_description=None,
                score_metadata=metadata,
                score_category=self._score_category,
                score_rationale=None,
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
                task=task.question,
            )
        ]
        request_response.scores = score
        return score

    async def score_prompts_with_tasks_batch_async(  # type: ignore[override]
        self,
        *,
        request_responses: Sequence[PromptRequestPiece],
        tasks: Sequence[QuestionAnsweringEntry],
        batch_size=10,
    ) -> list[Score]:
        if not tasks:
            raise ValueError("Tasks must be provided.")
        if len(request_responses) != len(tasks):
            raise ValueError(
                f"Number of tasks ({len(tasks)}) must match number of provided answers ({len(request_responses)})."
            )
        prompt_target = getattr(self, "_prompt_target", None)
        results = await batch_task_async(
            task_func=self.score_async,
            task_arguments=["request_response", "task"],
            prompt_target=prompt_target,
            batch_size=batch_size,
            items_to_batch=[request_responses, tasks],
        )
        return results

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
