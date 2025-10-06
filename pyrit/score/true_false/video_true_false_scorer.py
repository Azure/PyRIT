# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import (
    OR_,
    TrueFalseScoreAggregator,
)
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer
from pyrit.score.video_scorer import _BaseVideoScorer


class VideoTrueFalseScorer(TrueFalseScorer, _BaseVideoScorer):
    """
    A scorer that processes videos by extracting frames and scoring them using a true/false image scorer.

    The VideoTrueFalseScorer breaks down a video into frames and uses a true/false scoring mechanism.
    The frame scores are aggregated using a TrueFalseScoreAggregator (default: OR, meaning if any
    frame meets the objective, the entire video is scored as True).
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(supported_data_types=["video_path"])

    def __init__(
        self,
        *,
        image_capable_scorer: TrueFalseScorer,
        num_sampled_frames: Optional[int] = None,
        validator: Optional[ScorerPromptValidator] = None,
        score_aggregator: TrueFalseScoreAggregator = OR_,
    ) -> None:
        """
        Initialize the VideoTrueFalseScorer.

        Args:
            image_capable_scorer: A TrueFalseScorer capable of processing images.
            num_sampled_frames: Number of frames to extract from the video for scoring (default: 5).
            validator: Validator for the scorer. Defaults to video_path data type validator.
            score_aggregator: Aggregator for combining frame scores. Defaults to OR.
        """
        TrueFalseScorer.__init__(
            self, validator=validator or self._default_validator, score_aggregator=score_aggregator
        )
        _BaseVideoScorer.__init__(
            self, image_capable_scorer=image_capable_scorer, num_sampled_frames=num_sampled_frames
        )

    async def _score_piece_async(
        self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None
    ) -> list[Score]:
        """
        Score a single video piece by extracting frames and aggregating their scores.

        Args:
            request_piece: The prompt request piece containing the video.
            objective: Optional objective description for scoring.

        Returns:
            List containing a single aggregated score for the video.
        """

        # Get scores for all frames
        frame_scores = await self._score_frames_async(request_piece=request_piece, objective=objective)

        # Use the TrueFalseScoreAggregator to combine frame scores
        result = self._score_aggregator(frame_scores)

        # Get the ID from the request piece
        piece_id = request_piece.id if request_piece.id is not None else request_piece.original_prompt_id

        # Create the aggregated score using the aggregator result
        aggregate_score = Score(
            score_value=str(result.value).lower(),
            score_value_description=result.description,
            score_type="true_false",
            score_category=result.category,
            score_metadata=result.metadata,
            score_rationale=f"Video scored by analyzing {len(frame_scores)} frames.\n{result.rationale}",
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=piece_id,
            objective=objective,
        )

        return [aggregate_score]
