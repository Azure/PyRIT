# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unit.mocks import get_image_request_piece

from pyrit.models import PromptRequestPiece, Message
from pyrit.prompt_target import CrucibleTarget


@pytest.fixture
def crucible_target(patch_central_database) -> CrucibleTarget:
    return CrucibleTarget(endpoint="https://crucible", api_key="abc")


def test_crucible_initializes(crucible_target: CrucibleTarget):
    assert crucible_target


@pytest.mark.asyncio
async def test_crucible_validate_request_length(crucible_target: CrucibleTarget):
    request = Message(
        request_pieces=[
            PromptRequestPiece(role="user", conversation_id="123", original_value="test"),
            PromptRequestPiece(role="user", conversation_id="123", original_value="test2"),
        ]
    )
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await crucible_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_crucible_validate_prompt_type(crucible_target: CrucibleTarget):
    request_piece = get_image_request_piece()
    request = Message(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await crucible_target.send_prompt_async(prompt_request=request)
