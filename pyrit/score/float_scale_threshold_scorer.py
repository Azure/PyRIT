# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid

from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer import Scorer


class FloatScaleThresholdScorer(Scorer):
    """A scorer that leverages a scale scorer and applies a threshold to the score."""
    def __init__(self, *, memory: MemoryInterface, scorer: Scorer, threshold: float) -> None:
        self._scorer = scorer
        self._threshold = threshold
        self._memory = memory

        if not scorer._scorer_type == "float_scale":
            raise ValueError("The scorer must be a float scale scorer")

        if threshold <= 0 or threshold >= 1:
            raise ValueError("The threshold must be between 0 and 1")

        self._scorer_type = "true_false"
    
    async def score_async(self, request_response: PromptRequestPiece, *, task: str | None = None) -> list[Score]:
        """Scores the piece using the underlying float-scale scorer and thresholds the resulting score.
        
        Args:
            request_response (PromptRequestPiece): The piece to score.
            task (str): The task based on which the text should be scored.
            
        Returns:
            list[Score]: The scores.
        """
        scores = await self._scorer.score_async(request_response, task=task)
        for score in scores:
            score.score_value = score.score_value >= self._threshold
            score.score_type = self._scorer_type
            score.id = uuid.uuid4()
        self._memory.add_scores_to_memory(scores)
        return scores
    
    def validate(self, request_response: PromptRequestPiece, *, task: str | None = None) -> None:
        """Validates the request response for scoring."""
        self._scorer.validate(request_response, task=task)
