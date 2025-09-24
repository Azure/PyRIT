# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from pyrit.models import PromptRequestPiece, Score
from pyrit.score import Scorer, VideoScorer


def is_opencv_installed():
    try:
        import cv2  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


@pytest.fixture(autouse=True)
def video_converter_sample_video():
    # Create a sample video file
    video_path = "tests/unit/score/test_video.mp4"
    width, height = 512, 512
    if is_opencv_installed():
        import cv2  # noqa: F401

        # Create a video writer object
        video_encoding = cv2.VideoWriter_fourcc(*"mp4v")
        output_video = cv2.VideoWriter(video_path, video_encoding, 20, (width, height))
        # Create a few frames for video
        for i in range(10):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            processed_frame = cv2.flip(frame, 0)
            output_video.write(processed_frame)

        output_video.release()

    request_piece = PromptRequestPiece(
        role="user",
        original_value=video_path,
        converted_value=video_path,
        original_value_data_type="video_path",
        converted_value_data_type="video_path",
    )
    request_piece.id = None
    return request_piece


class MockScorer(Scorer):
    def __init__(self, scorer_type: str = "true_false"):
        self.scorer_type = scorer_type
        super().__init__()

    async def _score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        # For testing, always return a Score object with a predefined value
        if self.scorer_type == "true_false":
            return [
                Score(
                    score_type=self.scorer_type,
                    score_value="True",
                    score_rationale="Test true rationale",
                    score_category="test_category",
                    score_metadata="tf metadata",
                    score_value_description="test_tf_description",
                    prompt_request_response_id="test_id",
                )
            ]
        else:  # float_scale
            return [
                Score(
                    score_type=self.scorer_type,
                    score_value="0.8",
                    score_rationale="Test float rationale",
                    score_category="test_category",
                    score_metadata="float metadata",
                    score_value_description="test_float_description",
                    prompt_request_response_id="test_id",
                )
            ]

    async def score_image_async(self, image_path: str, *, task: Optional[str] = None) -> list[Score]:
        """Mock implementation for image scoring needed by video scoring"""
        # Create a mock PromptRequestPiece for the image
        request_piece = PromptRequestPiece(
            role="user",
            original_value=image_path,
            converted_value=image_path,
            original_value_data_type="image_path",
            converted_value_data_type="image_path",
        )
        return await self._score_async(request_piece, task=task)

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        pass


class MockVideoScorer(VideoScorer):
    """Test implementation of VideoScorer for video tests"""

    def __init__(self, image_scorer: Optional[MockScorer] = None, scorer_type: str = "true_false", num_frames: Optional[int] = 3):
        if not image_scorer:
            image_scorer = MockScorer(scorer_type=scorer_type)
        super().__init__(image_capable_scorer=image_scorer, num_frames=num_frames)

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        pass


@pytest.mark.asyncio
@pytest.mark.skipif(not is_opencv_installed(), reason="opencv is not installed")
async def test_extract_frames(video_converter_sample_video):
    """Test that frame extraction produces the expected number of frames"""
    import cv2

    scorer = MockVideoScorer()
    video_path = video_converter_sample_video.converted_value
    frame_paths = scorer._extract_frames(video_path=video_path)

    assert len(frame_paths) == scorer.num_frames, f"Expected {scorer.num_frames} frames, got {len(frame_paths)}"

    # Verify frames are valid images and cleanup
    for path in frame_paths:
        assert os.path.exists(path), f"Frame file {path} does not exist"
        img = cv2.imread(path)
        assert img is not None, f"Failed to read frame file {path}"
        assert img.shape == (512, 512, 3), f"Unexpected frame dimensions: {img.shape}"
        os.remove(path)  # Cleanup

    os.remove(video_path)


@pytest.mark.asyncio
@pytest.mark.skipif(not is_opencv_installed(), reason="opencv is not installed")
async def test_score_video_true_false(video_converter_sample_video):
    """Test video scoring with a true/false scorer"""
    scorer = MockVideoScorer(scorer_type="true_false")
    scores = await scorer._score_async(video_converter_sample_video)

    assert len(scores) == 1, "Expected one aggregated score"
    assert scores[0].score_type == "true_false"
    assert scores[0].score_value == "True"
    assert "3 out of 3 video frames meet objective" in scores[0].score_rationale
    assert "Test true rationale" in scores[0].score_rationale


@pytest.mark.asyncio
@pytest.mark.skipif(not is_opencv_installed(), reason="opencv is not installed")
async def test_score_video_float_scale(video_converter_sample_video):
    """Test video scoring with a float_scale scorer"""
    scorer = MockVideoScorer(scorer_type="float_scale")
    scores = await scorer._score_async(video_converter_sample_video)

    assert len(scores) == 1, "Expected one aggregated score"
    assert scores[0].score_type == "float_scale"
    assert scores[0].score_value == "0.8"  # Value from _score_async
    assert "Highest score across 3 frames: 0.8" in scores[0].score_rationale
    assert "Test float rationale" in scores[0].score_rationale


@pytest.mark.asyncio
@pytest.mark.skipif(not is_opencv_installed(), reason="opencv is not installed")
async def test_score_video_cleanup(video_converter_sample_video):
    """Test that temporary frame files are cleaned up after scoring"""
    scorer = MockVideoScorer()
    frame_paths = []

    # Mock _extract_frames to capture the frame paths
    original_extract_frames = scorer._extract_frames

    def mock_extract_frames(*args, **kwargs):
        nonlocal frame_paths
        frame_paths = original_extract_frames(*args, **kwargs)
        return frame_paths

    scorer._extract_frames = mock_extract_frames

    await scorer._score_async(video_converter_sample_video)

    # Verify all temporary files were cleaned up
    for path in frame_paths:
        assert not os.path.exists(path), f"Temporary frame file {path} was not cleaned up"


@pytest.mark.asyncio
@pytest.mark.skipif(not is_opencv_installed(), reason="opencv is not installed")
async def test_score_video_no_frames(video_converter_sample_video):
    """Test error handling when no frames can be extracted"""
    scorer = MockVideoScorer()

    # Mock _extract_frames to return empty list
    scorer._extract_frames = MagicMock(return_value=[])

    with pytest.raises(ValueError, match="No frames extracted from video for scoring."):
        await scorer._score_async(video_converter_sample_video)


@pytest.mark.asyncio
@pytest.mark.skipif(not is_opencv_installed(), reason="opencv is not installed")
async def test_score_video_no_scores(video_converter_sample_video):
    """Test error handling when frame scoring returns no scores"""
    image_scorer = MockScorer()

    # Mock score_image_batch_async to return empty list
    image_scorer.score_image_batch_async = AsyncMock(return_value=[])
    scorer = MockVideoScorer(image_scorer=image_scorer)

    with pytest.raises(ValueError, match="No scores returned for image frames extracted from video."):
        await scorer._score_async(video_converter_sample_video)


def test_aggregate_frame_scores_multiple_categories_true_false():
    """Test aggregating scores with multiple categories for true_false scorer"""
    scorer = MockVideoScorer(scorer_type="true_false")
    
    frame_scores = [
        Score(
            score_type="true_false",
            score_value="True",
            score_rationale="Frame 1 category A",
            score_category="category_A",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id1",
        ),
        Score(
            score_type="true_false",
            score_value="False",
            score_rationale="Frame 2 category A",
            score_category="category_A",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id2",
        ),
        Score(
            score_type="true_false",
            score_value="True",
            score_rationale="Frame 3 category B",
            score_category="category_B",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id3",
        ),
    ]
    
    aggregated = scorer._aggregate_frame_scores(frame_scores)
    
    assert len(aggregated) == 2, "Expected two aggregated scores (one per category)"
    
    # Check category A aggregation
    cat_a_score = next(s for s in aggregated if s.score_category == "category_A")
    assert cat_a_score.score_value == "True"
    assert "1 out of 2 video frames meet objective" in cat_a_score.score_rationale
    assert "Frame 1 category A" in cat_a_score.score_rationale
    
    # Check category B aggregation
    cat_b_score = next(s for s in aggregated if s.score_category == "category_B")
    assert cat_b_score.score_value == "True"
    assert "1 out of 1 video frames meet objective" in cat_b_score.score_rationale
    assert "Frame 3 category B" in cat_b_score.score_rationale


def test_aggregate_frame_scores_multiple_categories_float_scale():
    """Test aggregating scores with multiple categories for float_scale scorer"""
    scorer = MockVideoScorer(scorer_type="float_scale")
    
    frame_scores = [
        Score(
            score_type="float_scale",
            score_value="0.3",
            score_rationale="Frame 1 category X",
            score_category="category_X",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id1",
        ),
        Score(
            score_type="float_scale",
            score_value="0.9",
            score_rationale="Frame 2 category X",
            score_category="category_X",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id2",
        ),
        Score(
            score_type="float_scale",
            score_value="0.5",
            score_rationale="Frame 3 category Y",
            score_category="category_Y",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id3",
        ),
        Score(
            score_type="float_scale",
            score_value="0.7",
            score_rationale="Frame 4 category Y",
            score_category="category_Y",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id4",
        ),
    ]
    
    aggregated = scorer._aggregate_frame_scores(frame_scores)
    
    assert len(aggregated) == 2, "Expected two aggregated scores (one per category)"
    
    # Check category X aggregation (should pick 0.9)
    cat_x_score = next(s for s in aggregated if s.score_category == "category_X")
    assert cat_x_score.score_value == "0.9"
    assert "Highest score across 2 frames: 0.90" in cat_x_score.score_rationale
    assert "Frame 2 category X" in cat_x_score.score_rationale
    
    # Check category Y aggregation (should pick 0.7)
    cat_y_score = next(s for s in aggregated if s.score_category == "category_Y")
    assert cat_y_score.score_value == "0.7"
    assert "Highest score across 2 frames: 0.70" in cat_y_score.score_rationale
    assert "Frame 4 category Y" in cat_y_score.score_rationale


def test_aggregate_frame_scores_no_category():
    """Test aggregating scores when no categories are specified"""
    scorer = MockVideoScorer(scorer_type="true_false")
    
    frame_scores = [
        Score(
            score_type="true_false",
            score_value="True",
            score_rationale="Frame 1 no category",
            score_category=None,
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id1",
        ),
        Score(
            score_type="true_false",
            score_value="False",
            score_rationale="Frame 2 no category",
            score_category=None,
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id2",
        ),
    ]
    
    aggregated = scorer._aggregate_frame_scores(frame_scores)
    
    assert len(aggregated) == 1, "Expected one aggregated score when no categories"
    assert aggregated[0].score_value == "True"
    assert "1 out of 2 video frames meet objective" in aggregated[0].score_rationale
    assert aggregated[0].score_category is None


def test_aggregate_frame_scores_mixed_categories():
    """Test aggregating scores with some having categories and some without"""
    scorer = MockVideoScorer(scorer_type="true_false")
    
    frame_scores = [
        Score(
            score_type="true_false",
            score_value="True",
            score_rationale="Frame 1 with category",
            score_category="category_A",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id1",
        ),
        Score(
            score_type="true_false",
            score_value="False",
            score_rationale="Frame 2 no category",
            score_category=None,
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id2",
        ),
        Score(
            score_type="true_false",
            score_value="True",
            score_rationale="Frame 3 with category",
            score_category="category_A",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id3",
        ),
    ]
    
    aggregated = scorer._aggregate_frame_scores(frame_scores)

    # Check category A aggregation
    cat_a_score = next(s for s in aggregated if s.score_category == "category_A")
    assert cat_a_score.score_value == "True"
    assert "2 out of 2 video frames meet objective" in cat_a_score.score_rationale
    
    # Check aggregation for score without a category
    cat_none_score = next(s for s in aggregated if s.score_category == None)
    assert cat_none_score.score_value == "False"
    assert "Objective not met within 1 video frames" in cat_none_score.score_rationale
    assert "Frame 2 no category" in cat_none_score.score_rationale


def test_aggregate_frame_scores_all_false():
    """Test aggregating scores when all are False for true_false scorer"""
    scorer = MockVideoScorer(scorer_type="true_false")
    
    frame_scores = [
        Score(
            score_type="true_false",
            score_value="False",
            score_rationale="Frame 1 false",
            score_category="category_A",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id1",
        ),
        Score(
            score_type="true_false",
            score_value="False",
            score_rationale="Frame 2 false",
            score_category="category_A",
            score_metadata="metadata",
            score_value_description="description",
            prompt_request_response_id="id2",
        ),
    ]
    
    aggregated = scorer._aggregate_frame_scores(frame_scores)
    
    assert len(aggregated) == 1
    assert aggregated[0].score_value == "False"
    assert "Objective not met within 2 video frames" in aggregated[0].score_rationale
    assert "Frame 1 false" in aggregated[0].score_rationale  # Original rationale preserved

