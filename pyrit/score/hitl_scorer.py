# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import csv

from pathlib import Path
from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.models import PromptRequestPiece, Score
from pyrit.score import Scorer


class HITLScorer(Scorer):
    """
    Create scores from manual human input and adds them to the database.
    """

    def __init__(self,
                 *,
                 memory: MemoryInterface = None) -> None:
        self._memory = memory if memory else DuckDBMemory()

    def input_scores_from_csv(self, csv_file_path: Path) -> list[Score]:

        scores = []

        with open(csv_file_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                score = Score(
                    score_value=row["score_value"],
                    score_value_description=row.get("score_value_description", None),
                    score_type=row["score_type"],
                    score_category=row.get("score_category", None),
                    score_rationale=row.get("score_rationale", None),
                    score_metadata=row.get("score_metadata", None),
                    scorer_class_identifier=self.get_identifier(),
                    prompt_request_response_id=row.get("score_metadata", None),
                )
                scores.append(score)

        self._memory.add_scores_to_memory(scores=score)
        return scores

    async def score_async(self, request_response: PromptRequestPiece) -> list[Score]:

        await asyncio.sleep(0)

        self.validate(request_response)

        expected_output_substring_present = self._substring in request_response.converted_value

        score = [
            Score(
                score_value=str(expected_output_substring_present),
                score_value_description=None,
                score_metadata=None,
                score_type=self.scorer_type,
                score_category=self._category,
                score_rationale=None,
                scorer_class_identifier=self.get_identifier(),
                prompt_request_response_id=request_response.id,
            )
        ]

        self._memory.add_scores_to_memory(scores=score)
        return score
