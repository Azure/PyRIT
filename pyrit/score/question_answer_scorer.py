# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import textwrap
from typing import Generator, Tuple
from collections import defaultdict

from pydantic import BaseModel, ConfigDict

from pyrit.memory import CentralMemory
from pyrit.models import (
    QuestionAnsweringDataset, 
    QuestionAnsweringEntry,
    Score
)
from pyrit.score.scorer import Scorer
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from typing import Optional, Sequence
from pyrit.common.question_answer_helpers import construct_evaluation_prompt

class TextScoreResult(BaseModel):
    """Represents the result of scoring a text.

    Do not build on this class; this class needs to be rewritten.

    Parameters:
        provided_answer (str): The provided answer.
        correct_answer (str): The correct answer.
        is_correct (bool): Whether the provided answer is correct.
    """

    model_config = ConfigDict(extra="forbid")
    provided_answer: str
    correct_answer: str
    is_correct: bool

    def __str__(self) -> str:
        msg = f"Provided Answer: '{self.provided_answer}', "
        msg += f"Correct Answer: '{self.correct_answer}', "
        msg += f"Is Correct: '{self.is_correct}'"
        return msg


class QuestionAnswerScorer(Scorer):
    """A class that represents a question answering scorer.

    Parameters:
        dataset (QuestionAnsweringDataset): The dataset containing the questions and answers.
        evaluation_results (dict[QuestionAnsweringDataset, TextScoreResult]): A dictionary to store the evaluation
        results.

    """

    dataset: QuestionAnsweringDataset
    responses = list[PromptRequestResponse]

    def __init__(self, category: str = None) -> None:
        """
        Initializes the QuestionAnswerScorer object.
        """
        self._score_category = category
        self.scorer_type = "true_false"

    async def score_async(self, request_response: PromptRequestPiece, scored_question: QuestionAnsweringEntry, task: Optional[str] = None) -> Score:
        """
        Score the request_reponse using the QuestionAnsweringEntry
        and return a since Score object 30

        Args:
            answer (str): The answer given by the target
            task (QuestionAnsweringEntry): The entry containing the original prompt and the correct answer
        Returns:
            Score: A single Score object representing the result
        """
        answer = request_response.converted_value
        answer_correct = False
        try:
            # This is the case where the model response is an integer, which is the index of the answer.
            answer_index = int(answer)
            answer = scored_question.choices[answer_index].text
        except ValueError:
            # If the model response is not an integer, then the model might have returned the answer as a string
            pass

        metadata = {
            "question": str(scored_question.question),
            "correct_answer": str(scored_question.correct_answer), 
            "scored_answer": answer
        }

        answer_correct = str(scored_question.correct_answer) in answer

        score = Score(
            score_value=str(answer_correct),
            score_type=self.scorer_type,
            score_value_description=None,
            score_metadata=metadata,
            score_category=self._score_category,
            score_rationale=None,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.id,
            task=scored_question.question
        )

        return score

    async def evaluate(
        self, 
        dataset: QuestionAnsweringDataset,
        responses = Sequence[PromptRequestResponse]
    ) -> list[Score]:
        scores: list[Score] = []
        responses_batched = self._batch_responses_by_conversation_id(responses = responses)
        for question in dataset.questions:
            conversation_id = self._find_matching_conversation_from_question(question, responses_batched)
            scores.append(await self.score_async(responses_batched[conversation_id][3], question))
            responses_batched[conversation_id] = []
        return scores
    
    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        """
        Validates the request_response piece to score. Because some scorers may require
        specific PromptRequestPiece types or values.

        Args:
            request_response (PromptRequestPiece): The request response to be validated.
            task (str): The task based on which the text should be scored (the original attacker model's objective).
        """
        if (request_response.converted_value_data_type != "text"):
            raise ValueError("Question Answer Scorer only supports text data type")
    
    def _batch_responses_by_conversation_id(self, responses: Sequence[PromptRequestPiece]) -> defaultdict:
        responses_batched = defaultdict(list)
        for response in responses:
            responses_batched[response.conversation_id].append(response)
        return responses_batched
    
    def _find_matching_conversation_from_question(
        self,
        question: QuestionAnsweringEntry, 
        reseponses_batched: defaultdict
    ) -> str:
        question_prompt = construct_evaluation_prompt(question)
        for conversation in reseponses_batched:
            if reseponses_batched[conversation] != [] and question_prompt in reseponses_batched[conversation][2].converted_value:
                return conversation
