# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import pytest

from pyrit.prompt_target import GandalfLevel
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_target import GandalfTarget
from tests.mocks import get_sample_conversations


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def gandalf_target() -> GandalfTarget:
    return GandalfTarget(level=GandalfLevel.LEVEL_1)


def test_gandalf_initializes(gandalf_target: GandalfTarget):
    assert gandalf_target


@pytest.mark.asyncio
async def test_gandalf_validate_request_length(
    gandalf_target: GandalfTarget, sample_conversations: list[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await gandalf_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_gandalf_validate_prompt_type(
    gandalf_target: GandalfTarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await gandalf_target.send_prompt_async(prompt_request=request)
        

@pytest.mark.asyncio
async def test_gandalf_send_prompt_adds_memory_async(gandalf_target: GandalfTarget, sample_conversations: list[PromptRequestPiece]) -> None:

    mock_memory = MagicMock()
    gandalf_target._memory = mock_memory
    request_piece = sample_conversations[0]
    request_piece.conversation_id = str(uuid.uuid4())
    request = PromptRequestResponse(request_pieces=[request_piece])
    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        return_value = MagicMock()
        return_value.content = "input data"
        mock_request.return_value = return_value

        await gandalf_target.send_prompt_async(prompt_request=request)

        assert mock_memory.add_request_response_to_memory.called, "Request and Response need to be added to memory"
        assert mock_memory.add_response_entries_to_memory.called, "Request and Response need to be added to memory"
