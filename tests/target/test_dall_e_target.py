# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.prompt_target import DALLETarget
from pyrit.prompt_target.dall_e_target import SupportedDalleVersions

from tests.mocks import get_sample_conversations


@pytest.fixture
def image_target() -> DALLETarget:
    return DALLETarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
    )


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


def test_initialization_with_required_parameters(image_target: DALLETarget):
    assert image_target.deployment_name == "test"
    assert image_target.image_target is not None


def test_initialization_invalid_num_images():
    with pytest.raises(ValueError):
        DALLETarget(
            deployment_name="test",
            endpoint="test",
            api_key="test",
            dalle_version=SupportedDalleVersions.V3,
            num_images=3,
        )


@patch("pyrit.prompt_target.dall_e_target.DALLETarget._generate_images")
def test_send_prompt(mock_image, image_target, sample_conversations: list[PromptRequestPiece]):
    mock_image.return_value = {"Mock Image: ": "mock value"}
    resp = image_target.send_prompt(prompt_request=PromptRequestResponse([sample_conversations[0]]))
    assert resp


@patch("pyrit.prompt_target.dall_e_target.DALLETarget._generate_images_async")
@pytest.mark.asyncio
async def test_send_prompt_async(mock_image, image_target, sample_conversations: list[PromptRequestPiece]):
    mock_image.return_value = {"Mock Image: ": "mock value"}
    resp = await image_target.send_prompt_async(prompt_request=PromptRequestResponse(sample_conversations))
    assert resp
