# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import random
import tempfile
import uuid
from abc import ABC
from typing import Optional

from pyrit.models import MessagePiece, Score
from pyrit.score.scorer import Scorer

logger = logging.getLogger(__name__)


class _BaseVideoScorer(ABC):
    """
    Abstract base class for video scorers that process videos by extracting frames and scoring them.

    This class provides common functionality for extracting frames from videos and delegating
    scoring to an image-capable scorer. Concrete implementations handle aggregation logic
    specific to their scoring type (true/false or float scale).
    """

    _DEFAULT_VIDEO_FRAMES_SAMPLING_NUM = 5

    def __init__(
        self,
        *,
        image_capable_scorer: Scorer,
        num_sampled_frames: Optional[int] = None,
    ) -> None:
        """
        Initialize the base video scorer.

        Args:
            image_capable_scorer: A scorer capable of processing images that will be used to score
                individual video frames.
            num_sampled_frames: Number of frames to extract from the video for scoring (default: 5).

        Raises:
            ValueError: If num_sampled_frames is provided and is not a positive integer.
        """
        self.image_scorer = image_capable_scorer

        # Validate num_sampled_frames if provided
        if num_sampled_frames is not None and num_sampled_frames <= 0:
            raise ValueError("num_sampled_frames must be a positive integer")

        self.num_sampled_frames = (
            num_sampled_frames if num_sampled_frames is not None else self._DEFAULT_VIDEO_FRAMES_SAMPLING_NUM
        )

    async def _score_frames_async(self, *, message_piece: MessagePiece, objective: Optional[str] = None) -> list[Score]:
        """
        Extract frames from video and score them.

        Args:
            message_piece: The message piece containing the video.
            objective: Optional objective description for scoring.

        Returns:
            List of scores for the extracted frames.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If no frames are extracted from the video or if no scores are returned for the frames.
        """
        video_path = message_piece.converted_value

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Extract frames from video
        frames = self._extract_frames(video_path)
        if not frames:
            raise ValueError("No frames extracted from video for scoring.")

        # Score each frame
        objectives = [objective] * len(frames) if objective else None

        image_requests = []

        for frame in frames:
            # Convert original_prompt_id to UUID if it's a string
            original_prompt_id = message_piece.original_prompt_id
            if isinstance(original_prompt_id, str):
                original_prompt_id = uuid.UUID(original_prompt_id)

            piece = MessagePiece(
                original_value=message_piece.converted_value,
                role=message_piece.get_role_for_storage(),
                original_prompt_id=original_prompt_id,
                converted_value=frame,
                converted_value_data_type="image_path",
            )
            response = piece.to_message()
            image_requests.append(response)

        # Add the frame pieces to memory before scoring so that score references are valid
        from pyrit.memory import CentralMemory

        memory = CentralMemory.get_memory_instance()
        for request in image_requests:
            memory.add_message_to_memory(request=request)

        frame_scores = await self.image_scorer.score_prompts_batch_async(
            messages=image_requests, objectives=objectives, batch_size=len(frames)
        )

        if not frame_scores:
            raise ValueError("No scores returned for image frames extracted from video.")

        return frame_scores

    def _extract_frames(self, video_path: str) -> list[str]:
        """
        Extract a specified number of image frames from a video file.

        Args:
            video_path: The path to the video file.

        Returns:
            List of temporary file paths for the extracted frames.

        Raises:
            ModuleNotFoundError: If OpenCV is not installed.
        """
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError as e:
            logger.error("Could not import opencv. You may need to install it via 'pip install pyrit[opencv]'")
            raise e

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
                        logger.warning(f"Failed to read frame at index {frame_index}")
                        continue

                    # Create a temporary file for the frame
                    with tempfile.NamedTemporaryFile(suffix=f"_frame_{frame_index}.png", delete=False) as temp_file:
                        # Encode and write frame to temporary file
                        ret, _ = cv2.imencode(".png", frame)
                        if not ret:
                            logger.warning(f"Failed to encode frame at index {frame_index}")
                            continue

                        cv2.imwrite(temp_file.name, frame)
                        frame_paths.append(temp_file.name)
        finally:
            video_capture.release()

        return frame_paths
