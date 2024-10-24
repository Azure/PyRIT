# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv

from pathlib import Path
from typing import Optional
from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.models import PromptRequestPiece, Score
from pyrit.score import Scorer


class HumanInTheLoopScorer(Scorer):
    """
    Create scores from manual human input and adds them to the database.
    """

    def __init__(self, *, 
                    scorer: Scorer = None, # scorer we're using for the first score
                    re_scorer: list[Scorer] = None, # scorers we may use for re-scoring
                    memory: MemoryInterface = None) -> None:
         
        self._memory = memory if memory else DuckDBMemory()
        self._scorer = scorer
        self._re_scorer = re_scorer


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
        When the HumanInTheLoopScorer is used, user is given two options to choose from for each score:
        (1) Proceed with scoring the prompt as is
        (2) Manually modify the score & associated metadata

        Args:
            TODO
        Returns:
            TODO
        """

        # QUESTIONS: Want to ask user to score the prompt manually or re-score
        self.validate(request_response, task=task)
        
        
        user_input = ""
        
        """
        if no scorer, manually score it & ask them for all those fields
        """

        if self._scorer: # keep score, modify the score, or re-score
            # Score the response: 
            if not self._memory:
                await self._scorer.score_async(request_response=request_response, task=task)

            while user_input not in ["1", "2", "3"]:
                existing_scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[request_response.id])
                scored_prompt = self._memory.get_prompt_request_pieces_by_id(
                    prompt_ids=[existing_scores[0].prompt_request_response_id]
                )[0].converted_value
                user_input = input(
                    f"""Enter '1' to re-score the prompt. Enter 2 to manually edit the score. Enter 3 to proceed with scoring the prompt as is.
                    The prompt is: {scored_prompt}
                                    \nThe current score is: {existing_scores[0].score_value}
                                    \nThe current score category is: {existing_scores[0].score_category}
                                    \nThe current score value description is: {existing_scores[0].score_value_description}
                                    \nThe current score rationale is: {existing_scores[0].score_rationale}
                                    \nThe current score metadata is: {existing_scores[0].score_metadata}
                                    \nOr enter '2' to manually modify the score. Or enter '3' to re-score the prompt"""
                ).strip()
        
        else: # No score is given so you should manually score it
            score_value = input(f"""This prompt has not been scored yet, please manually score the prompt. 
                \nThe prompt is: {request_response.converted_value}
                \nEnter score value (e.g., 'True' for true_false or a value between '0.0' and '1.0 for float_scale): 
                """
            ).strip()

            score_type = "float_scale"
            if score_value.lower() == "true" or score_value.lower() == "false":
                score_type = "true_false"
            else:
                # This will throw an exception if the score is not a float
                value = float(score_value)
                if value < 0 or value > 1:
                    raise ValueError("Score value must be between 0 and 1 for float_scale scores")

            self._score_category = input("Enter score category (e.g., 'hate' or 'violence'): ").strip()
            score_value_description = self._optional_input("Enter score value description (optional): ")
            score_rationale = self._optional_input("Enter score rationale (optional): ")
            score_metadata = self._optional_input("Enter score metadata (optional): ")

            score = Score(
                score_value=score_value,
                score_value_description=score_value_description,
                score_type=score_type,  # type: ignore
                score_category=self._score_category,
                score_rationale=score_rationale,
                score_metadata=score_metadata,
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
                task=task,
            )
        

        if user_input == "2": # manually modify the score
            score_value = existing_scores[0].score_value
            change_value_input = input(f"""Re-scoring the prompt. The prompt is: {scored_prompt}
                                       \nThe previous score is: {score_value}
                                       \nDo you want to change the score value? Enter 1 for yes, 2 for no: """).strip()
            if change_value_input == "1":
                score_value = input(
                    "Enter modified score value (e.g., 'True' for true_false or a value between '0.0' and '1.0 for float_scale): "
                ).strip()
            
                score_type = "float_scale"
                if score_value.lower() == "true" or score_value.lower() == "false":
                    score_type = "true_false"
                else:
                    # This will throw an exception if the score is not a float
                    value = float(score_value)
                    if value < 0 or value > 1:
                        raise ValueError("Score value must be between 0 and 1 for float_scale scores")

            score_category = existing_scores[0].score_category
            change_value_input = input(f"""Re-scoring the prompt. The prompt is: {scored_prompt}
                                       \nThe previous score category is: {score_category}
                                       \nDo you want to change the score category? Enter 1 for yes, 2 for no: """).strip()
            
            score_value_description = existing_scores[0].score_value_description
            change_value_input = input(f"""Re-scoring the prompt. The prompt is: {scored_prompt}
                                       \nThe previous score value description is: {score_value_description}
                                       \nDo you want to change the score value description? Enter 1 for yes, 2 for no: """).strip()
            if change_value_input == "1":
                score_value_description = self._optional_input("Enter modified score value description (optional, press Enter to make empty): ").strip()
            
            score_rationale = existing_scores[0].score_rationale
            change_value_input = input(f"""Re-scoring the prompt. The prompt is: {scored_prompt}
                                    \nThe previous score rationale is: {score_rationale}
                                    \nDo you want to change the score rationale? Enter 1 for yes, 2 for no: """).strip()
            
            if change_value_input == "1":  
                score_rationale = self._optional_input("Enter score rationale (optional, press Enter to make empty): ").strip()

            score_metadata = existing_scores[0].score_metadata
            change_value_input = input(f"""Re-scoring the prompt. The prompt is: {scored_prompt}
                                    \nThe previous metadata is: {score_metadata}
                                    \nDo you want to change the score metadata? Enter 1 for yes, 2 for no: """).strip()
            
            if change_value_input == "1":
                score_metadata = self._optional_input("Enter new score metadata (optional, press Enter to make empty): ").strip()

            score = Score(
                score_value = score_value,
                score_value_description = score_value_description,
                score_type = score_type,  # type: ignore
                score_category = self._score_category,
                score_rationale = score_rationale,
                score_metadata = score_metadata,
                scorer_class_identifier = self.get_identifier(),
                prompt_request_response_id = request_response.id,
                task=task,
            )

        elif user_input == "3": # proceed with scoring the prompt as is
            score = Score(
                score_value = existing_scores[0].score_value,
                score_value_description = existing_scores[0].score_value_description,
                score_type = existing_scores[0].score_type,
                score_category = existing_scores[0].score_category,
                score_rationale = existing_scores[0].score_rationale,
                score_metadata=existing_scores[0].score_metadata,
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
                task=task,
            )

        elif user_input == "1":
            if not self._re_scorer:
                raise ValueError("No re-scorer provided")
            score = await self._re_scorer[0].score_async(request_response=request_response)
            

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def _optional_input(self, prompt):
        value = input(prompt)
        return None if value == "" else value

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if task:
            raise ValueError("This scorer does not support tasks")
