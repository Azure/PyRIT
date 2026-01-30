# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional

from pyrit.identifiers import ScorerIdentifier
from pyrit.models import MessagePiece, Score
from pyrit.score.float_scale.float_scale_score_aggregator import (
    FloatScaleAggregatorFunc,
    FloatScaleScorerByCategory,
)
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.score_aggregator_result import ScoreAggregatorResult
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.video_scorer import _BaseVideoScorer


class VideoFloatScaleScorer(
    FloatScaleScorer,
    _BaseVideoScorer,
):
    """
    A scorer that processes videos by extracting frames and scoring them using a float scale image scorer.

    The VideoFloatScaleScorer breaks down a video into frames and uses a float scale scoring mechanism.
    Frame scores are aggregated using a FloatScaleAggregatorFunc.

    By default, uses FloatScaleScorerByCategory.MAX which groups scores by category (useful for scorers like
    AzureContentFilterScorer that return multiple scores per frame). This returns one aggregated score
    per category (e.g., one for "Hate", one for "Violence", etc.).

    For scorers that return a single score per frame, or to combine all categories together,
    use FloatScaleScoreAggregator.MAX, FloatScaleScorerAllCategories.MAX, etc.

    Optionally, an audio_scorer can be provided to also score the video's audio track. When provided,
    the audio is extracted, transcribed, and scored. The audio scores are included in the aggregation.
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(supported_data_types=["video_path"])

    def __init__(
        self,
        *,
        image_capable_scorer: FloatScaleScorer,
        audio_scorer: Optional[FloatScaleScorer] = None,
        num_sampled_frames: Optional[int] = None,
        validator: Optional[ScorerPromptValidator] = None,
        score_aggregator: FloatScaleAggregatorFunc = FloatScaleScorerByCategory.MAX,
        ignore_objective_for_images: bool = False,
        ignore_objective_for_audio: bool = True,
    ) -> None:
        """
        Initialize the VideoFloatScaleScorer.

        Args:
            image_capable_scorer: A FloatScaleScorer capable of processing images.
            audio_scorer: Optional FloatScaleScorer for scoring the video's audio track.
                When provided, audio is extracted from the video, transcribed to text,
                and scored. The audio scores are aggregated with frame scores.
            num_sampled_frames: Number of frames to extract from the video for scoring (default: 5).
            validator: Validator for the scorer. Defaults to video_path data type validator.
            score_aggregator: Aggregator for combining frame scores. Defaults to FloatScaleScorerByCategory.MAX.
                Use FloatScaleScorerByCategory.MAX/AVERAGE/MIN for scorers that return multiple scores per frame
                (groups by category and returns one score per category).
                Use FloatScaleScorerAllCategories.MAX/AVERAGE/MIN to combine all scores regardless of category
                (returns single score with all categories combined).
                Use FloatScaleScoreAggregator.MAX/AVERAGE/MIN for simple aggregation preserving all categories
                (returns single score with all categories preserved).
            ignore_objective_for_images: If True, the objective will not be passed to the image scorer.
                Defaults to False (objective is passed to image scorer).
            ignore_objective_for_audio: If True, the objective will not be passed to the audio scorer.
                Defaults to True because video objectives typically describe visual content that
                doesn't apply to audio transcription.

        Raises:
            ValueError: If audio_scorer is provided and does not support audio_path data type.
        """
        FloatScaleScorer.__init__(self, validator=validator or self._default_validator)

        _BaseVideoScorer.__init__(
            self,
            image_capable_scorer=image_capable_scorer,
            num_sampled_frames=num_sampled_frames,
            ignore_objective_for_images=ignore_objective_for_images,
            ignore_objective_for_audio=ignore_objective_for_audio,
        )
        self._score_aggregator = score_aggregator

        if audio_scorer is not None:
            self._validate_audio_scorer(audio_scorer)
        self.audio_scorer = audio_scorer

    def _build_identifier(self) -> ScorerIdentifier:
        """
        Build the scorer evaluation identifier for this scorer.

        Returns:
            ScorerIdentifier: The identifier for this scorer.
        """
        sub_scorers = [self.image_scorer]
        if self.audio_scorer:
            sub_scorers.append(self.audio_scorer)

        return self._create_identifier(
            sub_scorers=sub_scorers,
            score_aggregator=self._score_aggregator.__name__,
            scorer_specific_params={
                "num_sampled_frames": self.num_sampled_frames,
                "has_audio_scorer": self.audio_scorer is not None,
                "ignore_objective_for_images": self.ignore_objective_for_images,
                "ignore_objective_for_audio": self.ignore_objective_for_audio,
            },
        )

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """
        Score a single video piece by extracting frames and optionally audio, then aggregating their scores.

        Args:
            message_piece: The message piece containing the video.
            objective: Optional objective description for scoring.

        Returns:
            List of aggregated scores for the video. Returns one score if using FloatScaleScoreAggregator,
            or multiple scores (one per category) if using FloatScaleScorerByCategory.
        """
        frame_scores = await self._score_frames_async(message_piece=message_piece, objective=objective)

        all_scores = list(frame_scores)
        audio_scored = False

        # Score audio if audio_scorer is provided
        if self.audio_scorer:
            audio_scores = await self._score_video_audio_async(
                message_piece=message_piece, audio_scorer=self.audio_scorer, objective=objective
            )
            if audio_scores:
                all_scores.extend(audio_scores)
                audio_scored = True

        # Get the ID from the message piece
        piece_id = message_piece.id if message_piece.id is not None else message_piece.original_prompt_id

        # Call the aggregator - all aggregators now return List[ScoreAggregatorResult]
        aggregator_results: List[ScoreAggregatorResult] = self._score_aggregator(all_scores)

        # Build rationale prefix
        rationale_prefix = f"Video scored by analyzing {len(frame_scores)} frames"
        if audio_scored:
            rationale_prefix += " and audio transcript"

        # Create Score objects from aggregator results
        aggregate_scores: List[Score] = []
        for result in aggregator_results:
            aggregate_score = Score(
                score_value=str(result.value),
                score_value_description=result.description,
                score_type="float_scale",
                score_category=result.category,
                score_metadata=result.metadata,
                score_rationale=f"{rationale_prefix}.\n{result.rationale}",
                scorer_class_identifier=self.get_identifier(),
                message_piece_id=piece_id,
                objective=objective,
            )
            aggregate_scores.append(aggregate_score)

        return aggregate_scores
