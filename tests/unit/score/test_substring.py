# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import MagicMock, patch

import pytest
from unit.mocks import get_image_request_piece

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece
from pyrit.score.substring_scorer import SubStringScorer


@pytest.fixture
def image_request_piece() -> PromptRequestPiece:
    return get_image_request_piece()


@pytest.mark.asyncio
async def test_substring_scorer_validate(image_request_piece: PromptRequestPiece):
    scorer = SubStringScorer(substring="test", category="new_category")
    with pytest.raises(ValueError, match="Expected text data type"):
        await scorer.score_async(image_request_piece)

    os.remove(image_request_piece.converted_value)


@pytest.mark.asyncio
@pytest.mark.parametrize("sub_string", ["I can't answer", "Not present"])
async def test_substring_scorer_score(sub_string: str, patch_central_database):
    full_text = "blah I can't answer that too"
    scorer = SubStringScorer(substring=sub_string, category="new_category")

    score = await scorer.score_text_async(full_text)

    assert len(score) == 1

    assert score[0].score_value == str(sub_string in full_text)
    assert score[0].score_type == "true_false"
    assert score[0].score_category == "new_category"
    assert score[0].prompt_request_response_id is None


@pytest.mark.asyncio
async def test_substring_scorer_adds_to_memory():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SubStringScorer(substring="string", category="new_category")
        await scorer.score_text_async(text="string")

        memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_substring_scorer_no_category():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SubStringScorer(substring="string")
        await scorer.score_text_async(text="string")

        memory.add_scores_to_memory.assert_called_once()
