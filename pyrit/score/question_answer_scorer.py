# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import textwrap
from typing import Generator, Tuple
from collections import defaultdict, deque

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
from pyrit.common.batch_helper import batch_task_async
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

    async def score_async(self, request_response: PromptRequestPiece, task: QuestionAnsweringEntry) -> Score:
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
            answer = task.choices[answer_index].text
        except ValueError:
            # If the model response is not an integer, then the model might have returned the answer as a string
            pass

        metadata = {
            "question": str(task.question),
            "correct_answer": str(task.correct_answer), 
            "scored_answer": answer
        }

        answer_correct = str(task.correct_answer) in answer

        score = Score(
            score_value=str(answer_correct),
            score_type=self.scorer_type,
            score_value_description=None,
            score_metadata=metadata,
            score_category=self._score_category,
            score_rationale=None,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.id,
            task=task.question
        )
        return score
    
    async def score_prompts_with_tasks_batch_async(
            self,
            *,
            request_responses:Sequence[PromptRequestPiece],
            tasks: Sequence[QuestionAnsweringEntry],
            batch_size = 10,
    ) -> list[Score]:
        if not tasks:
            raise ValueError("Tasks must be provided.")
        responses_batched = self._batch_responses_by_conversation_id(request_responses)
        request_responses = self._get_answers_in_order(tasks=tasks, responses_batched=responses_batched)
        if len(request_responses) != len(tasks):
            raise ValueError(f"The number of tasks ({len(tasks)}) must match the number of provided answers ({len(request_responses)}).")
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
        if (request_response.converted_value_data_type != "text"):
            raise ValueError("Question Answer Scorer only supports text data type")
    
    def _get_answers_in_order(
        self,
        tasks: Sequence[QuestionAnsweringEntry], 
        responses_batched: defaultdict
    ) -> list:
        if (len(tasks) != len(responses_batched)):
            raise ValueError("The number of questions must match the number of conversations")
        answers = []
        question_deque = deque(tasks)
        while len(question_deque):
            question = question_deque.popleft()
            matching_conversation = self._find_matching_conversation_from_question(question, responses_batched)
            answers.append(responses_batched[matching_conversation][3])
        return answers
    
    def _batch_responses_by_conversation_id(self, responses: Sequence[PromptRequestPiece]) -> defaultdict:
        responses_batched = defaultdict(list)
        for response in responses:
            responses_batched[response.conversation_id].append(response)
        return responses_batched
    
    def _find_matching_conversation_from_question(
        self,
        question: QuestionAnsweringEntry, 
        responses_batched: defaultdict
    ) -> str:
        question_prompt = construct_evaluation_prompt(question)
        for conversation in responses_batched:
            if responses_batched[conversation] != [] and question_prompt in responses_batched[conversation][2].converted_value:
                return conversation
