# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import random
import tempfile
import uuid
from collections import defaultdict
from typing import List, Optional

from pyrit.models import PromptRequestPiece, Score
from pyrit.score import Scorer

logger = logging.getLogger(__name__)

try:
    import cv2  # noqa: F401
except ModuleNotFoundError as e:
    logger.error("Could not import opencv. You may need to install it via 'pip install pyrit[opencv]'")
    raise e


class VideoScorer(Scorer):
    _DEFAULT_VIDEO_FRAMES_SAMPLING_NUM = 5
    """
    A scorer that processes videos by extracting frames and scoring them using an image-capable scorer.

    The VideoScorer breaks down a video into a specified number of frames and uses the provided
    image scorer to evaluate each frame. The final score is computed based on the composite scoring logic.
    """

    def __init__(self, image_capable_scorer: Scorer, num_sampled_frames: Optional[int] = None):
        """
        Initialize the VideoScorer.

        Args:
            image_capable_scorer: A scorer capable of processing images that will be used to score individual
                video frames
            num_sampled_frames: Number of frames to extract from the video for scoring (default: 5)
        """
        self.scorer_type = image_capable_scorer.scorer_type
        self.image_scorer = image_capable_scorer
        self.num_sampled_frames = num_sampled_frames or self._DEFAULT_VIDEO_FRAMES_SAMPLING_NUM

        if self.num_sampled_frames <= 0:
            raise ValueError("num_sampled_frames must be a positive integer")

    async def _score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> List[Score]:
        """
        Score a video by extracting frames and scoring each frame.

        Args:
            request_response: The prompt request piece containing the video
            task: Optional task description for scoring

        Returns:
            List of scores for each extracted frame
        """
        video_path = request_response.converted_value
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Extract frames from video
        frames = self._extract_frames(video_path)
        if not frames:
            raise ValueError("No frames extracted from video for scoring.")

        # Score each frame
        tasks = [task] * len(frames)
        frame_scores = await self.image_scorer.score_image_batch_async(
            image_paths=frames, tasks=tasks, batch_size=len(frames)
        )

        if not frame_scores:
            raise ValueError("No scores returned for image frames extracted from video.")

        # Clean up temporary frame files
        for path in frames:
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Error removing temporary frame file {path}: {e}")

        scores = self._aggregate_frame_scores(frame_scores)
        for score in scores:
            score.prompt_request_response_id = request_response.id

        return scores

    def _aggregate_frame_scores(self, frame_scores: list[Score]) -> List[Score]:
        """
        Aggregates a list of frame scores into a single score for the entire video.
        If there are multiple score categories within the list of scores provided,
        this function will return an aggregated score per-category.

        Args:
            frame_scores (list[Score]): The list of frame scores to aggregate.
        Returns:
            Score (list[Score]): The aggregated scores for the entire video.
        """

        # Group frame_scores by score_category (excluding None categories)
        scores_by_category = defaultdict(list)

        for score in frame_scores:
            if score.score_category:
                scores_by_category[score.score_category].append(score)
            else:
                scores_by_category[None].append(score)

        # Aggregate scores for each category
        if self.scorer_type == "true_false":
            return self._aggregate_true_false_scores(scores_by_category)  # type: ignore
        elif self.scorer_type == "float_scale":
            return self._aggregate_float_scale_scores(scores_by_category)  # type: ignore
        else:
            raise ValueError(f"Unsupported scorer type for video scoring: {self.scorer_type}")

    def _aggregate_true_false_scores(self, frame_scores: defaultdict[str, list[Score]]) -> List[Score]:
        """
        Aggregates a list of true/false frame scores into a single true/false score for the entire video.
        If any of the frame scores are True, the aggregate score is True.

        Args:
            frame_scores (list[Score]): The list of true/false frame scores to aggregate.
        Returns:
            Score: The aggregated true/false score for the entire video.
        """
        aggregated_scores = []
        aggregate_score = None
        original_rationale = None
        for category, category_scores in frame_scores.items():
            # For true_false scorers, if any of the scores are True, the aggregate score is True
            true_scores = [score for score in category_scores if score.get_value()]
            true_count = len(true_scores)

            # Update the rationale to reflect the aggregation logic
            if true_count > 0:
                original_rationale = "\nSee original score rationales for each frame as follows:\n* "
                original_rationale += "\n* ".join([score.score_rationale for score in true_scores])

                aggregate_score = true_scores[0]
                aggregate_score.score_rationale = (
                    f"{true_count} out of {len(category_scores)} video frames meet objective.\n"
                )
            else:
                aggregate_score = category_scores[0]
                original_rationale = aggregate_score.score_rationale

                aggregate_score.score_rationale = f"Objective not met within {len(category_scores)} video frames.\n"
            # Preserve the category in the aggregated score
            aggregate_score.score_category = category
            aggregate_score.id = uuid.uuid4()

            if original_rationale:
                aggregate_score.score_rationale += original_rationale

            aggregated_scores.append(aggregate_score)
        return aggregated_scores

    def _aggregate_float_scale_scores(self, frame_scores: defaultdict) -> List[Score]:
        """
        Aggregates a list of float scale frame scores into a single float scale score for the entire video.
        The aggregate score is the highest score value among the frame scores.

        Args:
            frame_scores (list[Score]): The list of float scale frame scores to aggregate.
        Returns:
            Score: The aggregated float scale score for the entire video.
        """
        aggregated_scores = []
        aggregate_score = None
        original_rationale = None
        for category, category_scores in frame_scores.items():
            # For float_scale scorers, use the highest score value as the aggregate score value
            aggregate_score = max(category_scores, key=lambda score: score.get_value())
            original_rationale = aggregate_score.score_rationale

            # Update the rationale to indicate this is the highest score
            highest_score = aggregate_score.get_value()
            aggregate_score.score_rationale = (
                f"Highest score across {len(category_scores)} frames: {highest_score:.2f}.\n"
            )

            # Preserve the category in the aggregated score
            aggregate_score.score_category = category
            aggregate_score.id = uuid.uuid4()

            if original_rationale:
                aggregate_score.score_rationale += original_rationale

            aggregated_scores.append(aggregate_score)

        return aggregated_scores

    def _extract_frames(self, video_path: str) -> list[str]:
        """
        Extracts a specified number of image frames from a video file and returns them as a list of (temp)
        image file paths.
        Args:
            video_path (str): The path to the video file.
        """

        frame_paths = []
        video_capture = cv2.VideoCapture(video_path)

        try:
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                # Choose up to num_sampled_frames random unique frame indices
                frame_indices = sorted(random.sample(range(total_frames), min(self.num_sampled_frames, total_frames)))
                for frame_index in frame_indices:
                    # Set the video position to the selected frame
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = video_capture.read()
                    if not ret:
                        print(f"Failed to read frame at index {frame_index}")
                        continue

                    # Create a temporary file for the frame
                    with tempfile.NamedTemporaryFile(suffix=f"_frame_{frame_index}.png", delete=False) as temp_file:
                        # Encode and write frame to temporary file
                        ret, _ = cv2.imencode(".png", frame)
                        if not ret:
                            print(f"Failed to encode frame at index {frame_index}")
                            continue

                        cv2.imwrite(temp_file.name, frame)
                        frame_paths.append(temp_file.name)
        finally:
            video_capture.release()
        return frame_paths

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> None:
        """
        Validate that the request contains a video file.

        Args:
            request_response: The prompt request piece to validate
            task: Optional task description
        """
        if request_response.converted_value_data_type != "video_path":
            raise ValueError(
                f"VideoScorer requires video_path data type, got {request_response.converted_value_data_type}"
            )

        if not request_response.converted_value:
            raise ValueError("Video path is empty")

        if not os.path.exists(request_response.converted_value):
            raise FileNotFoundError(f"Video file not found: {request_response.converted_value}")
