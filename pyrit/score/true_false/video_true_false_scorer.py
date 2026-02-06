# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.identifiers import ScorerIdentifier
from pyrit.models import MessagePiece, Score
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import TrueFalseScoreAggregator
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer
from pyrit.score.video_scorer import _BaseVideoScorer


class VideoTrueFalseScorer(TrueFalseScorer, _BaseVideoScorer):
    """
    A scorer that processes videos by extracting frames and scoring them using a true/false image scorer.

    Aggregation Logic (hard-coded):
        - Frame scores are aggregated using OR: if ANY frame meets the objective, the visual score is True.
        - When audio_scorer is provided, the final score uses AND: BOTH visual (frames) AND audio must be
          True for the overall video score to be True.

    This means:
        - Video-only scoring: True if any frame matches the objective
        - Video + Audio scoring: True only if both video frames AND audio transcript match their objectives
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(supported_data_types=["video_path"])

    def __init__(
        self,
        *,
        image_capable_scorer: TrueFalseScorer,
        audio_scorer: Optional[TrueFalseScorer] = None,
        num_sampled_frames: Optional[int] = None,
        validator: Optional[ScorerPromptValidator] = None,
        image_objective_template: Optional[str] = _BaseVideoScorer._DEFAULT_IMAGE_OBJECTIVE_TEMPLATE,
        audio_objective_template: Optional[str] = None,
    ) -> None:
        """
        Initialize the VideoTrueFalseScorer.

        Args:
            image_capable_scorer: A TrueFalseScorer capable of processing images.
            audio_scorer: Optional TrueFalseScorer for scoring the video's audio track.
                When provided, audio is extracted from the video and scored.
                The final score requires BOTH video frames AND audio to be True.
            num_sampled_frames: Number of frames to extract from the video for scoring (default: 5).
            validator: Validator for the scorer. Defaults to video_path data type validator.
            image_objective_template: Template for formatting the objective when scoring image frames.
                Use {objective} as placeholder for the actual objective. Set to None to not pass
                objective to image scorer. Defaults to a template that provides context about the
                video frame.
            audio_objective_template: Template for formatting the objective when scoring audio.
                Use {objective} as placeholder for the actual objective. Set to None to not pass
                objective to audio scorer. Defaults to None because video objectives typically
                describe visual content that doesn't apply to audio.

        Raises:
            ValueError: If audio_scorer is provided and does not support audio_path data type.
        """
        _BaseVideoScorer.__init__(
            self,
            image_capable_scorer=image_capable_scorer,
            num_sampled_frames=num_sampled_frames,
            image_objective_template=image_objective_template,
            audio_objective_template=audio_objective_template,
        )

        TrueFalseScorer.__init__(self, validator=validator or self._default_validator)

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
            scorer_specific_params={
                "num_sampled_frames": self.num_sampled_frames,
                "has_audio_scorer": self.audio_scorer is not None,
                "image_objective_template": self.image_objective_template,
                "audio_objective_template": self.audio_objective_template,
            },
        )

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """
        Score a single video piece by extracting frames and optionally audio, then aggregating their scores.

        Aggregation logic:
            - Frame scores are combined with OR (True if ANY frame matches)
            - If audio_scorer is provided, the final result is AND of (frame_result, audio_result)

        Args:
            message_piece: The message piece containing the video.
            objective: Optional objective description for scoring.

        Returns:
            List containing a single aggregated score for the video.
        """
        piece_id = message_piece.id if message_piece.id is not None else message_piece.original_prompt_id

        # Get scores for all frames and aggregate with OR (True if ANY frame matches)
        frame_scores = await self._score_frames_async(message_piece=message_piece, objective=objective)
        frame_result = TrueFalseScoreAggregator.OR(frame_scores)

        # Create a Score from the frame aggregation result
        frame_score = Score(
            score_value=str(frame_result.value).lower(),
            score_value_description=frame_result.description,
            score_type="true_false",
            score_category=frame_result.category,
            score_metadata=frame_result.metadata,
            score_rationale=f"Frames ({len(frame_scores)}): {frame_result.rationale}",
            scorer_class_identifier=self.get_identifier(),
            message_piece_id=piece_id,
            objective=objective,
        )

        # Score audio if audio_scorer is provided
        if self.audio_scorer:
            audio_scores = await self._score_video_audio_async(
                message_piece=message_piece, audio_scorer=self.audio_scorer, objective=objective
            )
            if audio_scores:
                # AND: both frame and audio must be true
                all_scores = [frame_score] + audio_scores
                final_result = TrueFalseScoreAggregator.AND(all_scores)
                return [
                    Score(
                        score_value=str(final_result.value).lower(),
                        score_value_description=final_result.description,
                        score_type="true_false",
                        score_category=final_result.category,
                        score_metadata=final_result.metadata,
                        score_rationale=final_result.rationale,
                        scorer_class_identifier=self.get_identifier(),
                        message_piece_id=piece_id,
                        objective=objective,
                    )
                ]

        # No audio: OR result from frames only
        return [frame_score]
