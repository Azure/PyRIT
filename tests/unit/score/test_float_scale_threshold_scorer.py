# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import Score
from pyrit.score import FloatScaleThresholdScorer


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("score_value", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.asyncio
async def test_float_scale_threshold_scorer_adds_to_memory(threshold, score_value):
    memory = MagicMock(MemoryInterface)

    scorer = AsyncMock()
    scorer.score_async = AsyncMock(
        return_value=[
            Score(
                score_value=str(score_value),
                score_type="float_scale",
                score_category=["mock category"],
                score_rationale="A mock rationale",
                score_metadata=None,
                message_piece_id=uuid.uuid4(),
                score_value_description="A mock description",
                id=uuid.uuid4(),
            )
        ]
    )
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

        float_scale_threshold_scorer = FloatScaleThresholdScorer(scorer=scorer, threshold=threshold)

        binary_score = (await float_scale_threshold_scorer.score_text_async(text="mock example"))[0]
        assert binary_score.score_value == str(score_value >= threshold)
        assert binary_score.score_type == "true_false"
        assert binary_score.score_value_description == "A mock description"

        memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_float_scale_threshold_scorer_returns_single_score_with_multi_category_scorer():
    """
    Test that FloatScaleThresholdScorer returns exactly one score even when the underlying scorer
    returns multiple.
    """

    memory = MagicMock(MemoryInterface)

    # Mock a scorer that returns multiple scores (like AzureContentFilterScorer)
    scorer = AsyncMock()
    prompt_id = uuid.uuid4()
    scorer.score_async = AsyncMock(
        return_value=[
            Score(
                score_value="0.2",
                score_type="float_scale",
                score_category=["Hate"],
                score_rationale="Hate rationale",
                score_metadata={"azure_severity": 2},
                message_piece_id=prompt_id,
                score_value_description="",
                id=uuid.uuid4(),
            ),
            Score(
                score_value="0.0",
                score_type="float_scale",
                score_category=["Violence"],
                score_rationale="Violence rationale",
                score_metadata={"azure_severity": 0},
                message_piece_id=prompt_id,
                score_value_description="",
                id=uuid.uuid4(),
            ),
            Score(
                score_value="0.8",
                score_type="float_scale",
                score_category=["Sexual"],
                score_rationale="Sexual rationale",
                score_metadata={"azure_severity": 6},
                message_piece_id=prompt_id,
                score_value_description="",
                id=uuid.uuid4(),
            ),
        ]
    )

    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        float_scale_threshold_scorer = FloatScaleThresholdScorer(scorer=scorer, threshold=0.5)

        result_scores = await float_scale_threshold_scorer.score_text_async(text="mock example")

        # Should return exactly one score
        assert len(result_scores) == 1

        binary_score = result_scores[0]
        # With MAX aggregator (default), should take max value (0.8) which is >= 0.5
        assert binary_score.get_value() is True
        assert binary_score.score_type == "true_false"

        # Verify memory was called once with a single score
        memory.add_scores_to_memory.assert_called_once()
        added_scores = memory.add_scores_to_memory.call_args[1]["scores"]
        assert len(added_scores) == 1
