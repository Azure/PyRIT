# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.models import PromptRequestPiece, Score
from pyrit.score.composite_scorer import CompositeScorer
from pyrit.score.score_aggregator import AND_, MAJORITY_, OR_
from pyrit.score.scorer import Scorer


class MockScorer(Scorer):
    """A mock scorer for testing purposes."""

    def __init__(self, score_value: bool, score_rationale: str):
        self.scorer_type = "true_false"
        self._score_value = score_value
        self._score_rationale = score_rationale

    async def score_async(self, request_response: PromptRequestPiece, *, task: str = None) -> list[Score]:
        return [
            Score(
                score_value=str(self._score_value),
                score_value_description=None,
                score_type=self.scorer_type,
                score_category=None,
                score_metadata=None,
                score_rationale=self._score_rationale,
                scorer_class_identifier={"name": "MockScorer"},
                prompt_request_response_id=request_response.id,
                task=task,
            )
        ]

    def validate(self, request_response: PromptRequestPiece, *, task: str = None) -> None:
        pass


@pytest.fixture
def mock_request():
    return PromptRequestPiece(role="user", original_value="test content", conversation_id="test-conv", sequence=1)


@pytest.fixture
def true_scorer():
    return MockScorer(True, "This is a true score")


@pytest.fixture
def false_scorer():
    return MockScorer(False, "This is a false score")


@pytest.mark.asyncio
async def test_composite_scorer_and_all_true(mock_request, true_scorer):
    scorer = CompositeScorer(aggregator=AND_, scorers=[true_scorer, true_scorer])

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is True
    assert "All constituent scorers returned True" in scores[0].score_rationale


@pytest.mark.asyncio
async def test_composite_scorer_and_one_false(mock_request, true_scorer, false_scorer):
    scorer = CompositeScorer(aggregator=AND_, scorers=[true_scorer, false_scorer])

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is False
    assert "At least one constituent scorer returned False" in scores[0].score_rationale


@pytest.mark.asyncio
async def test_composite_scorer_or_all_false(mock_request, false_scorer):
    scorer = CompositeScorer(aggregator=OR_, scorers=[false_scorer, false_scorer])

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is False
    assert "All constituent scorers returned False" in scores[0].score_rationale


@pytest.mark.asyncio
async def test_composite_scorer_or_one_true(mock_request, true_scorer, false_scorer):
    scorer = CompositeScorer(aggregator=OR_, scorers=[true_scorer, false_scorer])

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is True
    assert "At least one constituent scorer returned True" in scores[0].score_rationale


@pytest.mark.asyncio
async def test_composite_scorer_majority_true(mock_request, true_scorer, false_scorer):
    scorer = CompositeScorer(aggregator=MAJORITY_, scorers=[true_scorer, true_scorer, false_scorer])

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is True
    assert "A strict majority of constituent scorers returned True" in scores[0].score_rationale


@pytest.mark.asyncio
async def test_composite_scorer_majority_false(mock_request, true_scorer, false_scorer):
    scorer = CompositeScorer(aggregator=MAJORITY_, scorers=[true_scorer, false_scorer, false_scorer])

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is False
    assert "A strict majority of constituent scorers did not return True" in scores[0].score_rationale


def test_composite_scorer_invalid_scorer_type():
    class InvalidScorer(Scorer):
        def __init__(self):
            self.scorer_type = "invalid_type"

        async def score_async(self, request_response: PromptRequestPiece, *, task: str = None) -> list[Score]:
            return []

        def validate(self, request_response: PromptRequestPiece, *, task: str = None) -> None:
            pass

    with pytest.raises(ValueError, match="All scorers must be true_false scorers"):
        CompositeScorer(aggregator=AND_, scorers=[InvalidScorer()])


@pytest.mark.asyncio
async def test_composite_scorer_with_task(mock_request, true_scorer):
    scorer = CompositeScorer(aggregator=AND_, scorers=[true_scorer])

    task = "test task"
    scores = await scorer.score_async(mock_request, task=task)
    assert len(scores) == 1
    assert scores[0].task == task


def test_composite_scorer_empty_scorers_list():
    """Test that CompositeScorer raises an exception when given an empty list of scorers."""
    with pytest.raises(ValueError, match="At least one scorer must be provided"):
        CompositeScorer(aggregator=AND_, scorers=[])
