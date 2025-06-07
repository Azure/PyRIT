# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import MutableSequence

import pytest
from unit.mocks import get_sample_conversations

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import CrucibleTarget


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def crucible_target() -> CrucibleTarget:
    return CrucibleTarget(endpoint="https://crucible", api_key="abc")


def test_crucible_initializes(crucible_target: CrucibleTarget):
    assert crucible_target


@pytest.mark.asyncio
async def test_crucible_validate_request_length(
    crucible_target: CrucibleTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await crucible_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_crucible_validate_prompt_type(
    crucible_target: CrucibleTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await crucible_target.send_prompt_async(prompt_request=request)
