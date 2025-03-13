# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import MutableSequence

import pytest
from unit.mocks import get_sample_conversations

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import GandalfLevel, GandalfTarget


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def gandalf_target() -> GandalfTarget:
    return GandalfTarget(level=GandalfLevel.LEVEL_1)


def test_gandalf_initializes(gandalf_target: GandalfTarget):
    assert gandalf_target


@pytest.mark.asyncio
async def test_gandalf_validate_request_length(
    gandalf_target: GandalfTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await gandalf_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_gandalf_validate_prompt_type(
    gandalf_target: GandalfTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await gandalf_target.send_prompt_async(prompt_request=request)
