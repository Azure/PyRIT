# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import csv

from pathlib import Path
from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.models import PromptRequestPiece, Score
from pyrit.score import Scorer


class HumanInTheLoopScorer(Scorer):
    """
    Create scores from manual human input and adds them to the database.
    """

    def __init__(self, *, memory: MemoryInterface = None) -> None:
        self._memory = memory if memory else DuckDBMemory()

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
                )
                scores.append(score)

        # This is post validation, so the scores should be okay and normalized
        self._memory.add_scores_to_memory(scores=scores)
        return scores

    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:

        await asyncio.sleep(0)

        print("Scoring the following:")
        print(request_response)

        score_value = input(
            "Enter score value (e.g., 'True' for true_false or a value between '0.0'\
                             and '1.0 for float_scale): "
        )

        score_type = "float_scale"
        if score_value.lower() == "true" or score_value.lower() == "false":
            score_type = "true_false"
        else:
            # This will throw an exception if the score is not a float
            value = float(score_value)
            if value < 0 or value > 1:
                raise ValueError("Score value must be between 0 and 1 for float_scale scores")

        score_category = input("Enter score category (e.g., 'hate' or 'violence'): ")
        score_value_description = self._optional_input("Enter score value description (optional): ")
        score_rationale = self._optional_input("Enter score rationale (optional): ")
        score_metadata = self._optional_input("Enter score metadata (optional): ")

        score = Score(
            score_value=score_value,
            score_value_description=score_value_description,
            score_type=score_type,  # type: ignore
            score_category=score_category,
            score_rationale=score_rationale,
            score_metadata=score_metadata,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.id,
        )

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def _optional_input(self, prompt):
        value = input(prompt)
        return None if value == "" else value

    def validate(self, request_response: PromptRequestPiece):
        pass
