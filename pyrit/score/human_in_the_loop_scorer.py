# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
from pathlib import Path
from typing import Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer import Scorer


class HumanInTheLoopScorer(Scorer):
    """
    Create scores from manual human input and adds them to the database.

    Parameters:
        scorer (Scorer): The scorer to use for the initial scoring.
        re_scorers (list[Scorer]): The scorers to use for re-scoring.
    """

    def __init__(self, *, scorer: Scorer = None, re_scorers: list[Scorer] = None) -> None:
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

    def score_prompt_manually(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Manually score the prompt

        Args:
            request_response (PromptRequestPiece): The prompt request piece to score.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            list of scores
        """
        self.validate(request_response, task=task)

        score_value = ""
        score_category = ""
        while not score_value or not score_category:
            if not score_category:
                score_category = self._get_user_input("Please enter score category (e.g., 'hate' or 'violence').")

            if not score_value:
                message = f"""This prompt has not been scored yet, please manually score the prompt.
                The prompt is: {request_response.converted_value}\n
                Please enter a score value
                (e.g., 'True' for true_false or a value between '0.0' and '1.0 for float_scale): """

                score_value = self._get_user_input(message)
        score_type = self._get_score_type(score_value)
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

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Score the prompt with a human in the loop.

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
        new_scores = []
        original_prompt = request_response.converted_value

        if self._scorer:
            user_input = ""
            # Score the response using provided scorer
            scores = await self._scorer.score_async(request_response=request_response, task=task)

            if not self._re_scorers:
                user_choice_list = ["1", "2"]
            else:
                user_choice_list = ["1", "2", "3"]

            for existing_score in scores:
                while user_input not in user_choice_list:
                    if user_choice_list == ["1", "2"]:  # no re-scorers provided
                        message = f"""
                        Enter '1' to proceed with scoring the prompt as is.
                        Enter '2' to manually edit the score.\n
                        The prompt is: {original_prompt}\n
                        Current score details:
                            - Score Value: {existing_score.score_value}
                            - Score Category: {existing_score.score_category}
                            - Score Value Description: {existing_score.score_value_description}
                            - Score Rationale: {existing_score.score_rationale}
                            - Score Metadata: {existing_score.score_metadata}"""

                    else:
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
            return self.score_prompt_manually(request_response, task=task)

        for user_input in input_list:
            if user_input == "1":  # proceed with scoring the prompt as is
                new_scores.append(existing_score)

            elif user_input == "2":  # manually modify the score
                score = self.edit_score(existing_score, original_prompt, request_response, task)
                new_scores.append(score)

            elif user_input == "3":
                return await self.rescore(request_response, task=task)

        return new_scores

    def _get_user_input(self, message) -> str:
        try:
            import tkinter as tk
            from tkinter import simpledialog
        except ImportError as e:
            print(
                "To use HumanInTheLoopScorer, you need to install tkinter. "
                "See https://stackoverflow.com/a/74607246 for more information."
            )
            raise e
        root = tk.Tk()
        root.withdraw()
        user_input = simpledialog.askstring("Score Prompt", message).strip()
        root.destroy()
        return user_input.strip()

    def _get_score_type(self, score_value: str) -> str:
        """
        Checks score value to make sure it is a valid float scale score or true/false score and returns the score type.

        Parameters:
            score_value (str): The score value to check.

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

    def edit_score(
        self, existing_score: Score, original_prompt: str, request_response: PromptRequestPiece, task: Optional[str]
    ) -> Score:
        """
        Edit an existing score.

        Parameters:
            existing_score (Score): The existing score to edit.
            original_prompt (str): The original prompt.
            request_response (PromptRequestPiece): The request response to score.
            task (str): The task based on which the text should be scored (the original attacker model's objective).

        Returns:
            new score after all changes
        """
        # Update each field
        value_description = "(e.g., 'True' for true_false or a value between '0.0' and '1.0 for float_scale)"
        score_value = self.get_modified_value(
            original_prompt=original_prompt,
            score_value=existing_score.score_value,
            extra_value_description=value_description,
            field_name="score value",
        )
        score_type = self._get_score_type(score_value)

        value_description = "(e.g., 'hate' or 'violence')"
        score_category = self.get_modified_value(
            original_prompt=original_prompt,
            score_value=existing_score.score_category,
            extra_value_description=value_description,
            field_name="score category",
        )

        score_value_description = self.get_modified_value(
            original_prompt=original_prompt,
            score_value=existing_score.score_value_description,
            field_name="score value description",
        )

        score_rationale = self.get_modified_value(
            original_prompt=original_prompt, score_value=existing_score.score_rationale, field_name="score rationale"
        )

        score_metadata = self.get_modified_value(
            original_prompt=original_prompt, score_value=existing_score.score_metadata, field_name="score metadata"
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
        return score

    def get_modified_value(
        self,
        original_prompt: str,
        score_value: str,
        field_name: str,
        extra_value_description: str = "",
    ) -> str:
        """
        Get the modified value for the score.

        Args:
            original_prompt (str): The original prompt.
            score_value (str): The existing value in the Score object.
            field_name (str): The name of the field to change.
            extra_value_description (Optional str): Extra information to show user describing the score value.

        Returns:
            The value after modification or the original value if the user does not want to change it.
        """
        formatted_message = f"""Re-scoring the prompt. The prompt is: {original_prompt}
        The previous {field_name.capitalize()} is {score_value}.
        Do you want to change the previous value? Enter 1 for yes, 2 for no: """
        change_value_input = self._get_user_input(formatted_message)
        if extra_value_description:
            user_change_message = f"Enter modified {field_name.capitalize()}\n\
            {extra_value_description}\nOr press Enter to skip and keep old value: "
        else:
            user_change_message = (
                f"Enter modified {field_name.capitalize()} \n Or press Enter to skip and keep old value: "
            )
        return score_value if change_value_input == "2" else self._get_user_input(user_change_message)

    async def rescore(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        scorers_str = str([scorer.__class__.__name__ for scorer in self._re_scorers])
        scorer_index = -1
        message = f"""The available scorers are {scorers_str}. \
            Enter the index of the scorer you would like to run on the input (0 to {len(self._re_scorers)-1})"""
        while not 0 <= scorer_index < len(self._re_scorers):
            scorer_index = int(self._get_user_input(message))
        re_scorer = self._re_scorers[scorer_index]
        return await re_scorer.score_async(request_response=request_response, task=task)

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        pass
