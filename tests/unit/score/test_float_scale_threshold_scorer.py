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
    scorer.scorer_type = "float_scale"
    scorer.score_async = AsyncMock(
        return_value=[
            Score(
                score_value=str(score_value),
                score_type="float_scale",
                score_category="mock category",
                score_rationale="A mock rationale",
                score_metadata=None,
                prompt_request_response_id=uuid.uuid4(),
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
