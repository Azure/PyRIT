import os
import tempfile
from typing import List, Optional
import random
import logging
import uuid
from pyrit.models import PromptRequestPiece, Score
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class VideoScorer(Scorer):
    """
    A scorer that processes videos by extracting frames and scoring them using an image-capable scorer.
    
    The VideoScorer breaks down a video into a specified number of frames and uses the provided
    image scorer to evaluate each frame. The final score is computed based on the composite scoring logic.
    """
    
    def __init__(self, image_capable_scorer: Scorer, num_frames: Optional[int] = None):
        """
        Initialize the VideoScorer.
        
        Args:
            image_capable_scorer: A scorer capable of processing images that will be used to score individual video frames
            num_frames: Number of frames to extract from the video for scoring (default: 10)
        """
        self.scorer_type = image_capable_scorer.scorer_type
        self.image_scorer = image_capable_scorer

        if not num_frames:
            self.num_frames = 5
        else:
            self.num_frames = num_frames

        if self.num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
    
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
            image_paths=frames, tasks=tasks, batch_size=self.num_frames
        )
        
        if not frame_scores:
            raise ValueError("No scores returned for image frames extracted from video.")

        # Clean up temporary frame files
        for path in frames:
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Error removing temporary frame file {path}: {e}")

        score = self._aggregate_frame_scores(frame_scores)
        score.prompt_request_response_id = request_response.id

        return [score]
    
    def _aggregate_frame_scores(self, frame_scores: list[Score]) -> Score:
        """
        Aggregates a list of frame scores into a single score for the entire video.
        Args:
            frame_scores (list[Score]): The list of frame scores to aggregate.
        Returns:
            Score (list[Score]): The aggregated scores for the entire video.
        """
        # Aggregate frame scores into one score for the entire video
        aggregate_score = None
        original_rationale = None

        # For true_false scorers, if any of the scores are True, the aggregate score is True
        if self.scorer_type == "true_false":
            true_scores = [score for score in frame_scores if score.get_value()]
            true_count = len(true_scores)

            # Update the rationale to reflect the aggregation logic
            if true_count > 0:
                original_rationale = "\nSee original score rationales for each frame as follows:\n* "
                original_rationale += "\n* ".join([score.score_rationale for score in true_scores])

                aggregate_score = true_scores[0]
                aggregate_score.score_rationale = (
                    f"{true_count} out of {len(frame_scores)} video frames meet objective.\n"
                )
            else:
                aggregate_score = frame_scores[0]
                original_rationale = aggregate_score.score_rationale

                aggregate_score.score_rationale = f"Objective not met within {len(frame_scores)} video frames.\n"

        # For float_scale scorers, use the highest score value as the aggregate score value
        elif self.scorer_type == "float_scale":
            aggregate_score = max(frame_scores, key=lambda score: score.get_value())
            original_rationale = aggregate_score.score_rationale

            # Update the rationale to indicate this is the highest score
            highest_score = aggregate_score.get_value()
            aggregate_score.score_rationale = f"Highest score across {len(frame_scores)} frames: {highest_score:.2f}.\n"

        else:
            raise ValueError(f"Unsupported scorer type for video scoring: {self.scorer_type}")

        if original_rationale:
            # Append the original rationale to the updated rationale
            aggregate_score.score_rationale += original_rationale

        # Set the Score ID
        aggregate_score.id = uuid.uuid4()

        return aggregate_score
    
    def _extract_frames(self, video_path: str) -> list[str]:
        """
        Extracts a specified number of image frames from a video file and returns them as a list of (temp)
        image file paths.
        Args:
            video_path (str): The path to the video file.
            num_frames (int): The number of image frames to extract from the video.
        """
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError as e:
            logger.error("Could not import opencv. You may need to install it via 'pip install pyrit[opencv]'")
            raise e

        video_capture = cv2.VideoCapture(video_path)
        frame_paths = []
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            # Choose up to num_frames random unique frame indices
            frame_indices = sorted(random.sample(range(total_frames), min(self.num_frames, total_frames)))
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
            raise ValueError(f"VideoScorer requires video_path data type, got {request_response.converted_value_data_type}")
        
        if not request_response.converted_value:
            raise ValueError("Video path is empty")
        
        if not os.path.exists(request_response.converted_value):
            raise FileNotFoundError(f"Video file not found: {request_response.converted_value}")