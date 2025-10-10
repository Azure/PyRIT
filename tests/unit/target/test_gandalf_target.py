# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unit.mocks import get_image_request_piece

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import GandalfLevel, GandalfTarget


@pytest.fixture
def gandalf_target(patch_central_database) -> GandalfTarget:
    return GandalfTarget(level=GandalfLevel.LEVEL_1)


def test_gandalf_initializes(gandalf_target: GandalfTarget):
    assert gandalf_target


@pytest.mark.asyncio
async def test_gandalf_validate_request_length(gandalf_target: GandalfTarget):
    request = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="user", conversation_id="123", original_value="test"),
            PromptRequestPiece(role="user", conversation_id="123", original_value="test2"),
        ]
    )
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await gandalf_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_gandalf_validate_prompt_type(gandalf_target: GandalfTarget):
    request = PromptRequestResponse(request_pieces=[get_image_request_piece()])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await gandalf_target.send_prompt_async(prompt_request=request)
