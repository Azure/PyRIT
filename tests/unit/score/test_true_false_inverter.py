# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import MagicMock, patch

import pytest
from unit.mocks import get_image_message_piece

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import MessagePiece
from pyrit.score import SubStringScorer, TrueFalseInverterScorer


@pytest.fixture
def image_message_piece() -> MessagePiece:
    return get_image_message_piece()


@pytest.mark.asyncio
async def test_inverter_scorer_validate(image_message_piece: MessagePiece):
    sub_scorer = SubStringScorer(substring="test", categories=["new_category"])
    scorer = TrueFalseInverterScorer(scorer=sub_scorer)

    request = image_message_piece.to_message()

    with pytest.raises(ValueError, match="There are no valid pieces to score"):
        await scorer.score_async(request)

    os.remove(image_message_piece.converted_value)


@pytest.mark.asyncio
@pytest.mark.parametrize("sub_string", ["I can't answer", "Not present"])
async def test_substring_scorer_score(patch_central_database, sub_string: str):
    full_text = "blah I can't answer that too"

    sub_scorer = SubStringScorer(substring=sub_string, categories=["new_category"])
    scorer = TrueFalseInverterScorer(scorer=sub_scorer)

    score = await scorer.score_text_async(full_text)

    assert len(score) == 1

    # score_value should be the opposite of substring scorer since results are inverted
    assert score[0].score_value != str(sub_string in full_text)
    assert score[0].score_type == "true_false"
    assert score[0].score_category == ["new_category"]
    assert score[0].prompt_request_response_id is None


@pytest.mark.asyncio
async def test_substring_scorer_adds_to_memory():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SubStringScorer(substring="string", categories=["new_category"])
        await scorer.score_text_async(text="string")

        memory.add_scores_to_memory.assert_called_once()
