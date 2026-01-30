# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from pyrit.identifiers import ScorerIdentifier
from pyrit.models import MessagePiece, Score
from pyrit.score.audio_transcript_scorer import _BaseAudioTranscriptScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import (
    TrueFalseAggregatorFunc,
    TrueFalseScoreAggregator,
)
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer
from pyrit.score.video_scorer import _BaseVideoScorer


class VideoTrueFalseScorer(TrueFalseScorer, _BaseVideoScorer):
    """
    A scorer that processes videos by extracting frames and scoring them using a true/false image scorer.

    The VideoTrueFalseScorer breaks down a video into frames and uses a true/false scoring mechanism.
    The frame scores are aggregated using a TrueFalseAggregatorFunc (default: TrueFalseScoreAggregator.OR,
    meaning if any frame meets the objective, the entire video is scored as True).

    Optionally, an audio_scorer can be provided to also score the video's audio track. When provided,
    the audio is extracted, transcribed, and scored. The audio score is then aggregated with the
    frame scores using the same aggregation function.
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(supported_data_types=["video_path"])

    def __init__(
        self,
        *,
        image_capable_scorer: TrueFalseScorer,
        audio_scorer: Optional[TrueFalseScorer] = None,
        num_sampled_frames: Optional[int] = None,
        validator: Optional[ScorerPromptValidator] = None,
        score_aggregator: TrueFalseAggregatorFunc = TrueFalseScoreAggregator.OR,
    ) -> None:
        """
        Initialize the VideoTrueFalseScorer.

        Args:
            image_capable_scorer: A TrueFalseScorer capable of processing images.
            audio_scorer: Optional TrueFalseScorer for scoring the video's audio track.
                When provided, audio is extracted from the video, transcribed to text,
                and scored. The audio score is aggregated with frame scores.
            num_sampled_frames: Number of frames to extract from the video for scoring (default: 5).
            validator: Validator for the scorer. Defaults to video_path data type validator.
            score_aggregator: Aggregator for combining frame scores. Defaults to TrueFalseScoreAggregator.OR.

        Raises:
            ValueError: If audio_scorer is provided and does not support audio_path data type.
        """
        _BaseVideoScorer.__init__(
            self, image_capable_scorer=image_capable_scorer, num_sampled_frames=num_sampled_frames
        )

        TrueFalseScorer.__init__(
            self, validator=validator or self._default_validator, score_aggregator=score_aggregator
        )

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
            },
        )

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """
        Score a single video piece by extracting frames and optionally audio, then aggregating their scores.

        Args:
            message_piece: The message piece containing the video.
            objective: Optional objective description for scoring.

        Returns:
            List containing a single aggregated score for the video.
        """
        # Get scores for all frames
        frame_scores = await self._score_frames_async(message_piece=message_piece, objective=objective)

        all_scores = list(frame_scores)
        audio_scored = False

        # Score audio if audio_scorer is provided
        if self.audio_scorer:
            audio_score = await self._score_video_audio_async(message_piece=message_piece, objective=objective)
            if audio_score:
                all_scores.append(audio_score)
                audio_scored = True

        # Use the TrueFalseAggregatorFunc to combine all scores (frames + audio)
        result = self._score_aggregator(all_scores)

        # Get the ID from the message piece
        piece_id = message_piece.id if message_piece.id is not None else message_piece.original_prompt_id

        # Build rationale
        rationale_parts = [f"Video scored by analyzing {len(frame_scores)} frames"]
        if audio_scored:
            rationale_parts.append("and audio transcript")
        rationale_parts.append(f".\n{result.rationale}")

        # Create the aggregated score using the aggregator result
        aggregate_score = Score(
            score_value=str(result.value).lower(),
            score_value_description=result.description,
            score_type="true_false",
            score_category=result.category,
            score_metadata=result.metadata,
            score_rationale="".join(rationale_parts),
            scorer_class_identifier=self.get_identifier(),
            message_piece_id=piece_id,
            objective=objective,
        )

        return [aggregate_score]

    async def _score_video_audio_async(
        self, *, message_piece: MessagePiece, objective: Optional[str] = None
    ) -> Optional[Score]:
        """
        Extract and score audio from the video.

        Args:
            message_piece: The message piece containing the video.
            objective: Optional objective description for scoring.

        Returns:
            Score for the audio content, or None if audio extraction/scoring fails.
        """
        import os
        import uuid

        from pyrit.memory import CentralMemory

        video_path = message_piece.converted_value

        # Use _BaseAudioTranscriptScorer's static method to extract audio
        audio_path = _BaseAudioTranscriptScorer.extract_audio_from_video(video_path)
        if not audio_path:
            return None

        try:
            # Create a message piece for the audio
            original_prompt_id = message_piece.original_prompt_id
            if isinstance(original_prompt_id, str):
                original_prompt_id = uuid.UUID(original_prompt_id)

            audio_piece = MessagePiece(
                original_value=audio_path,
                role=message_piece.get_role_for_storage(),
                original_prompt_id=original_prompt_id,
                converted_value=audio_path,
                converted_value_data_type="audio_path",
            )

            audio_message = audio_piece.to_message()

            # Add to memory
            memory = CentralMemory.get_memory_instance()
            memory.add_message_to_memory(request=audio_message)

            # Score the audio using the audio_scorer
            # We pass objective=None because when used within a video scorer,
            # the audio should be evaluated independently using only its true_description,
            # not the overall video objective (which describes visual elements)
            if self.audio_scorer is None:
                return None
            audio_scores = await self.audio_scorer.score_prompts_batch_async(
                messages=[audio_message],
                objectives=None,  # Audio uses only its own true_description, not video objective
                batch_size=1,
            )

            return audio_scores[0] if audio_scores else None

        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            pass
