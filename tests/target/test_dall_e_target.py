# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch
import uuid

import pytest

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models import PromptRequestResponse
from pyrit.prompt_target import DALLETarget

from tests.mocks import get_sample_conversations


@pytest.fixture
def dalle_target() -> DALLETarget:
    return DALLETarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
    )


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


def test_initialization_with_required_parameters(dalle_target: DALLETarget):
    assert dalle_target
    assert dalle_target.deployment_name == "test"
    assert dalle_target.image_target is not None


def test_initialization_invalid_num_images():
    with pytest.raises(ValueError):
        DALLETarget(
            deployment_name="test",
            endpoint="test",
            api_key="test",
            dalle_version="dall-e-3",
            num_images=3,
        )


@patch("pyrit.prompt_target.dall_e_target.DALLETarget._generate_images_async")
@pytest.mark.asyncio
async def test_send_prompt_async(mock_image, dalle_target: DALLETarget, sample_conversations: list[PromptRequestPiece]):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())
    mock_image.return_value = {"data": [{"b64_json": "mock_json"}]}
    resp = await dalle_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
    assert resp


@pytest.mark.asyncio
async def test_dalle_validate_request_length(dalle_target: DALLETarget, sample_conversations: list[PromptRequestPiece]):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await dalle_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_dalle_validate_prompt_type(dalle_target: DALLETarget, sample_conversations: list[PromptRequestPiece]):
    request_piece = sample_conversations[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await dalle_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_dalle_validate_previous_conversations(
    dalle_target: DALLETarget, sample_conversations: list[PromptRequestPiece]
):
    request_piece = sample_conversations[0]
    dalle_target._memory.add_request_response_to_memory(request=PromptRequestResponse(request_pieces=[request_piece]))
    request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(ValueError, match="This target only supports a single turn conversation."):
        await dalle_target.send_prompt_async(prompt_request=request)
