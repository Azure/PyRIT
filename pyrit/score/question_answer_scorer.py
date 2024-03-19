# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Generator, Tuple
from pydantic import BaseModel, ConfigDict
from pyrit.models import QuestionAnsweringEntry, QuestionAnsweringDataset
import textwrap


class TextScoreResult(BaseModel):
    """Represents the result of scoring a text.

    Attributes:
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


class AggregateScoringResults(BaseModel):
    """
    Represents the aggregate scoring results.

    Attributes:
        total_failed (int): The total number of failed questions.
        total_passed (int): The total number of passed questions.
        total_questions_count (int): The total number of questions.

    """

    model_config = ConfigDict(extra="forbid")
    total_failed: int = 0
    total_passed: int = 0
    total_questions_count: int = 0

    def __str__(self) -> str:
        msg = f"Total Passed: {self.total_passed}, \n"
        msg += f"Total Failed: {self.total_failed}, \n"
        msg += f"Total Question Count: {self.total_questions_count}"
        return msg

    def add_result(self, result: TextScoreResult):
        """
        Adds the results of a single scoring to the aggregate results.

        Args:
            result (TextScoreResult): The scoring results to be added.
        """
        if result.is_correct:
            self.total_passed += 1
        else:
            self.total_failed += 1
        self.total_questions_count = self.total_failed + self.total_passed

    @classmethod
    def from_results_list(cls, results_list: list[TextScoreResult]) -> AggregateScoringResults:
        """
        Creates an instance of AggregateScoringResults from a list of TextScoreResult.

        Args:
            results_list (list[TextScoreResult]): The list of scoring results.

        Returns:
            AggregateScoringResults: An instance of AggregateScoringResults with aggregated results from the list.
        """
        instance = cls()
        for results in results_list:
            instance.add_result(results)
        return instance


class QuestionAnswerScorer:
    """A class that represents a question answering scorer.

    Attributes:
        dataset (QuestionAnsweringDataset): The dataset containing the questions and answers.
        evaluation_results (dict[QuestionAnsweringDataset, TextScoreResult]): A dictionary to store the evaluation
        results.

    """

    dataset: QuestionAnsweringDataset
    evaluation_results: dict[QuestionAnsweringEntry, TextScoreResult]

    def __init__(self, dataset: QuestionAnsweringDataset):
        """
        Initializes the QuestionAnswerScorer object.

        Args:
            dataset (QuestionAnsweringDataset): The dataset containing the questions and answers.
        """
        self.dataset = dataset
        self.evaluation_results = {}

    def _construct_evaluation_prompt(self, *, entry: QuestionAnsweringEntry) -> str:
        available_choices = ""
        for c in entry.choices:
            available_choices += f"index={c.index}, value={c.text}\n"

        return textwrap.dedent(
            f"""\
            Questions:
            {entry.question}

            Choices:
            {available_choices}

            Answer:

            """
        )

    def get_next_question_prompt_pair(self) -> Generator[Tuple[QuestionAnsweringEntry, str], None, None]:
        """
        Generates the next question-prompt pair from the dataset.

        Yields:
            A tuple containing the next question-prompt pair.
                - The first element is a QuestionAnsweringEntry object representing the question.
                - The second element is a string representing the prompt that should be asked.

        """
        for entry in self.dataset.questions:
            prompt = self._construct_evaluation_prompt(entry=entry)
            yield entry, prompt

    def score_question(self, question: QuestionAnsweringEntry, answer: str) -> TextScoreResult:
        """Scores the provided answer for a given question.

        Args:
            question (QuestionAnsweringEntry): The question to be scored.
            answer (str): The answer provided by the model.

        Returns:
            TextScoreResult: The score result for the question and answer pair.
        """
        valid_answer_found = False
        is_answer_correct = False
        predicted_answer_content_by_index = ""
        try:
            # This is the case where the model response is an integer, which is the index of the answer.
            predicted_answer_index = int(answer)
            predicted_answer_content_by_index = question.choices[predicted_answer_index].text
            valid_answer_found = True
        except ValueError:
            # If the model response is not an integer, then the model might have returned the answer as a string
            pass

        if str(question.correct_answer) in answer:
            # Try to see if the model contains that answer as a substring.
            # If the correct answer is a substring of the model response, then mark the result as correct.
            is_answer_correct = True
            valid_answer_found = True

        if is_answer_correct:
            score_result = TextScoreResult(
                correct_answer=str(question.correct_answer),
                provided_answer=answer,
                is_correct=True,
            )
        elif valid_answer_found:
            score_result = TextScoreResult(
                correct_answer=str(question.correct_answer),
                # Note, we might want to return the full answer here, not just the index.
                provided_answer=predicted_answer_content_by_index,
                is_correct=predicted_answer_content_by_index == question.correct_answer,
            )
        else:
            score_result = TextScoreResult(
                correct_answer=str(question.correct_answer),
                provided_answer=answer,
                is_correct=False,
            )
        self.evaluation_results[question] = score_result
        return score_result
