# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import tkinter as tk
from tkinter import simpledialog

from pathlib import Path
from typing import Optional
from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.models import PromptRequestPiece, Score
from pyrit.score import Scorer


class HumanInTheLoopScorer(Scorer):
    """
    Create scores from manual human input and adds them to the database.

    Attributes:
        scorer (Scorer): The scorer to use for the initial scoring.
        re_scorers (list[Scorer]): The scorers to use for re-scoring.
        memory (MemoryInterface): The memory interface
    """

    def __init__(
        self, *, scorer: Scorer = None, re_scorers: list[Scorer] = None, memory: MemoryInterface = None
    ) -> None:

        self._memory = memory if memory else DuckDBMemory()
        self._scorer = scorer
        self._re_scorers = re_scorers

    def import_scores_from_csv(self, csv_file_path: Path | str) -> list[Score]:

        scores = []

        with open(csv_file_path, newline="") as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                score = Score(
                    score_value=row["score_value"],
                    score_value_description=row.get("score_value_description", None),
                    score_type=row["score_type"],  # type: ignore
                    score_category=row.get("score_category", None),
                    score_rationale=row.get("score_rationale", None),
                    score_metadata=row.get("score_metadata", None),
                    scorer_class_identifier=self.get_identifier(),
                    prompt_request_response_id=row["prompt_request_response_id"],
                    task=row.get("task", None),
                )
                scores.append(score)

        # This is post validation, so the scores should be okay and normalized
        self._memory.add_scores_to_memory(scores=scores)
        return scores

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        When the HumanInTheLoopScorer is used, user is given three options to choose from for each score:
        (1) Proceed with scoring the prompt as is
        (2) Manually modify the score & associated metadata
            If the user chooses to manually modify the score,
            they are prompted to enter the new score value, score category,
            score value description, score rationale, and score metadata
        (3) Re-score the prompt
            If the user chooses to re-score the prompt,
            they are prompted to select a re-scorer from the list of re-scorers provided

        If the user initializes this scorer without a scorer, they will be prompted to manually score the prompt.
        Args:
            request_response (PromptRequestPiece): The prompt request piece to score.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list[Score]: The request_response scored.
        """

        self.validate(request_response, task=task)

        input_list = []
        original_prompt = request_response.converted_value
        new_scores = []

        if self._scorer:
            user_input = ""
            # Score the response using provided scorer
            scores = await self._scorer.score_async(request_response=request_response, task=task)
            for existing_score in scores:
                while user_input not in ["1", "2", "3"]:
                    message = f"""
                        Enter '1' to proceed with scoring the prompt as is.
                        Enter '2' to manually edit the score.
                        Enter '3' to re-score the prompt.\n
                        The prompt is: {original_prompt}\n
                        Current score details:
                            - Score Value: {existing_score.score_value}
                            - Score Category: {existing_score.score_category}
                            - Score Value Description: {existing_score.score_value_description}
                            - Score Rationale: {existing_score.score_rationale}
                            - Score Metadata: {existing_score.score_metadata}"""

                    user_input = self._get_user_input(message)

                input_list.append(user_input)
        else:  # No scorer is given so you should manually score it

            score_value = ""
            while not score_value:
                message = f"""This prompt has not been scored yet, please manually score the prompt.
                The prompt is: {original_prompt}\n
                Please enter a score value
                (e.g., 'True' for true_false or a value between '0.0' and '1.0 for float_scale): """

                score_value = self._get_user_input(message)
            score_type = self._get_score_type(score_value)

            score_category = ""
            while not score_category:
                score_category = self._get_user_input("Please enter score category (e.g., 'hate' or 'violence').")
            score_value_description = self._get_user_input(
                "Enter score value description (optional, press 'Enter' to skip): "
            )
            score_rationale = self._get_user_input("Enter score rationale (optional, press 'Enter' to skip): ")
            score_metadata = self._get_user_input("Enter score metadata (optional, press 'Enter' to skip): ")

            score = Score(
                score_value=score_value,
                score_value_description=score_value_description,
                score_type=score_type,  # type: ignore
                score_category=score_category,
                score_rationale=score_rationale,
                score_metadata=score_metadata,
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
                task=task,
            )

            return [score]

        for user_input in input_list:
            if user_input == "1":  # proceed with scoring the prompt as is
                return [existing_score]

            elif user_input == "2":  # manually modify the score

                # First, get the existing score value and type
                score_value = existing_score.score_value
                score_type = existing_score.score_type
                message = f"""Re-scoring the prompt. The prompt is: {original_prompt}
                    The previous score is: {score_value}
                    Do you want to change the score value? Enter 1 for yes, 2 for no: """
                change_value_input = self._get_user_input(message)
                if change_value_input == "1":
                    score_value = self._get_user_input(
                        "Enter modified score value \
                        (e.g., 'True' for true_false or a value between '0.0' and '1.0 for float_scale) \
                         press Enter to skip and keep old value: "
                    )
                    score_type = self._get_score_type(score_value)

                # Update each of the other fields
                score_category = existing_score.score_category

                message = f"""Re-scoring the prompt. The prompt is: {original_prompt}
                    The previous score category is: {score_category}
                    Do you want to change the score category? Enter 1 for yes, 2 for no: """
                change_value_input = self._get_user_input(message)
                if change_value_input == "1":
                    score_category = self._get_user_input(
                        "Enter modified score category (e.g., 'hate' or 'violence'): \
                                        press Enter to skip and keep old value:"
                    )

                score_value_description = existing_score.score_value_description
                message = f"""Re-scoring the prompt. The prompt is: {original_prompt}
                    The previous score value description is: {score_value_description}
                    Do you want to change the score value description? Enter 1 for yes, 2 for no: """
                change_value_input = self._get_user_input(message)
                if change_value_input == "1":
                    score_value_description = self._get_user_input(
                        "Enter modified score value description, press Enter to skip and keep old value: "
                    )

                score_rationale = existing_score.score_rationale
                message = f"""Re-scoring the prompt. The prompt is: {original_prompt}
                    The previous score rationale is: {score_rationale}
                    Do you want to change the score rationale? Enter 1 for yes, 2 for no: """
                change_value_input = self._get_user_input(message)
                if change_value_input == "1":
                    score_rationale = self._get_user_input(
                        "Enter score rationale, press Enter to skip and keep old value: "
                    )

                score_metadata = existing_score.score_metadata
                message = f"""Re-scoring the prompt. The prompt is: {original_prompt}
                    The previous metadata is: {score_metadata}
                    Do you want to change the score metadata? Enter 1 for yes, 2 for no: """
                change_value_input = self._get_user_input(message)
                if change_value_input == "1":
                    score_metadata = self._get_user_input(
                        "Enter new score metadata, press Enter to skip and keep old value: "
                    )

                score = Score(
                    score_value=score_value,
                    score_value_description=score_value_description,
                    score_type=score_type,  # type: ignore
                    score_category=score_category,
                    score_rationale=score_rationale,
                    score_metadata=score_metadata,
                    scorer_class_identifier=self.get_identifier(),
                    prompt_request_response_id=request_response.id,
                    task=task,
                )
                new_scores.append(score)

            elif user_input == "3":
                if not self._re_scorers:
                    raise ValueError("No re-scorer provided")

                scorers_str = str([scorer.__class__.__name__ for scorer in self._re_scorers])
                scorer_index = -1
                message = f"""The available scorers are {scorers_str}.
                    Enter the index of the scorer you would like to run on the input (0 to {len(self._re_scorers)-1})"""
                while not 0 <= scorer_index < len(self._re_scorers):
                    scorer_index = int(self._get_user_input(message))
                re_scorer = self._re_scorers[scorer_index]
                return await re_scorer.score_async(request_response=request_response, task=task)

        return new_scores

    def _get_user_input(self, message) -> str:
        root = tk.Tk()
        root.withdraw()
        user_input = simpledialog.askstring("Score Prompt", message).strip()
        root.destroy()
        return user_input.strip()

    def _get_score_type(self, score_value: str) -> str:
        """
        Checks score value to make sure it is a valid float scale score or true/false score and returns the score type.

        Parameters: score_value (str): The score value to check.
        Returns:
            str: The score type which is either true/false or float_scale.
        """
        if score_value.lower() == "true" or score_value.lower() == "false":
            return "true_false"
        else:
            # This will throw an exception if the score is not a float
            value = float(score_value)
            if value < 0 or value > 1:
                raise ValueError("Score value must be between 0 and 1 for float_scale scores")
        return "float_scale"

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if task:
            raise ValueError("This scorer does not support tasks")
