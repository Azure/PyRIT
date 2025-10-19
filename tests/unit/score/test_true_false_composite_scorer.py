# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
from unittest.mock import MagicMock

import pytest

from pyrit.memory.central_memory import CentralMemory
from pyrit.models import MessagePiece, Score
from pyrit.score import (
    FloatScaleScorer,
    TrueFalseCompositeScorer,
    TrueFalseScoreAggregator,
    TrueFalseScorer,
)


class MockScorer(TrueFalseScorer):
    """A mock scorer for testing purposes."""

    def _score_aggregator(self, score_list):
        # Use the AND aggregator from the TrueFalseScoreAggregator class
        return TrueFalseScoreAggregator.AND(score_list)

    def __init__(self, score_value: bool, score_rationale: str, aggregator=None):
        self.scorer_type = "true_false"
        self._score_value = score_value
        self._score_rationale = score_rationale
        self._validator = MagicMock()
        self.aggregator = aggregator

    async def _score_piece_async(
        self, message_piece: MessagePiece, *, objective: Optional[str] = None
    ) -> list[Score]:
        return [
            Score(
                score_value=str(self._score_value),
                score_value_description="",
                score_type=self.scorer_type,
                score_category=[],
                score_metadata=None,
                score_rationale=self._score_rationale,
                scorer_class_identifier={"name": "MockScorer"},
                prompt_request_response_id=str(message_piece.id),
                objective=str(objective),
            )
        ]


@pytest.fixture
def mock_request(patch_central_database):
    memory = CentralMemory.get_memory_instance()
    request = MessagePiece(role="user", original_value="test content", conversation_id="test-conv", sequence=1)
    memory.add_message_pieces_to_memory(message_pieces=[request])
    return request.to_message()


@pytest.fixture
def true_scorer(patch_central_database):
    return MockScorer(True, "This is a true score")


@pytest.fixture
def false_scorer(patch_central_database):
    return MockScorer(False, "This is a false score")


@pytest.mark.asyncio
async def test_composite_scorer_and_all_true(mock_request, true_scorer):
    scorer = TrueFalseCompositeScorer(aggregator=TrueFalseScoreAggregator.AND, scorers=[true_scorer, true_scorer])

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is True
    assert "This is a true score" in scores[0].score_rationale
    assert "All constituent scorers returned True in an AND composite scorer." in scores[0].score_value_description


@pytest.mark.asyncio
async def test_composite_scorer_and_one_false(mock_request, true_scorer, false_scorer):
    scorer = TrueFalseCompositeScorer(aggregator=TrueFalseScoreAggregator.AND, scorers=[true_scorer, false_scorer])

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is False
    assert "This is a false score" in scores[0].score_rationale
    assert "This is a true score" in scores[0].score_rationale


@pytest.mark.asyncio
async def test_composite_scorer_or_all_false(mock_request, false_scorer):
    scorer = TrueFalseCompositeScorer(aggregator=TrueFalseScoreAggregator.OR, scorers=[false_scorer, false_scorer])

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is False
    assert "This is a false score" in scores[0].score_rationale
    assert "All constituent scorers returned False in an OR composite scorer." in scores[0].score_value_description


@pytest.mark.asyncio
async def test_composite_scorer_or_one_true(mock_request, true_scorer, false_scorer):
    scorer = TrueFalseCompositeScorer(aggregator=TrueFalseScoreAggregator.OR, scorers=[true_scorer, false_scorer])

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is True
    assert "This is a true score" in scores[0].score_rationale


@pytest.mark.asyncio
async def test_composite_scorer_majority_true(mock_request, true_scorer, false_scorer):
    scorer = TrueFalseCompositeScorer(
        aggregator=TrueFalseScoreAggregator.MAJORITY, scorers=[true_scorer, true_scorer, false_scorer]
    )

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is True
    assert "This is a true score" in scores[0].score_rationale
    assert (
        "A strict majority of constituent scorers returned True in a MAJORITY composite scorer."
        in scores[0].score_value_description
    )


@pytest.mark.asyncio
async def test_composite_scorer_majority_false(mock_request, true_scorer, false_scorer):
    scorer = TrueFalseCompositeScorer(
        aggregator=TrueFalseScoreAggregator.MAJORITY, scorers=[true_scorer, false_scorer, false_scorer]
    )

    scores = await scorer.score_async(mock_request)
    assert len(scores) == 1
    assert scores[0].get_value() is False
    assert "This is a true score" in scores[0].score_rationale
    assert "This is a false score" in scores[0].score_rationale


def test_composite_scorer_invalid_scorer_type():

    class InvalidScorer(FloatScaleScorer):
        def __init__(self):
            self._validator = MagicMock()

        async def _score_piece_async(
            self, message_piece: MessagePiece, *, objective: Optional[str] = None
        ) -> list[Score]:
            return []

    with pytest.raises(ValueError, match="All scorers must be true_false scorers"):
        TrueFalseCompositeScorer(aggregator=TrueFalseScoreAggregator.AND, scorers=[InvalidScorer()])  # type: ignore


@pytest.mark.asyncio
async def test_composite_scorer_with_task(mock_request, true_scorer):
    scorer = TrueFalseCompositeScorer(aggregator=TrueFalseScoreAggregator.AND, scorers=[true_scorer])

    task = "test task"
    scores = await scorer.score_async(mock_request, objective=task)
    assert len(scores) == 1
    assert scores[0].objective == task


def test_composite_scorer_empty_scorers_list():
    """Test that CompositeScorer raises an exception when given an empty list of scorers."""
    with pytest.raises(ValueError, match="At least one scorer must be provided"):
        TrueFalseCompositeScorer(aggregator=TrueFalseScoreAggregator.AND, scorers=[])
