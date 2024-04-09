# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from pyrit.prompt_target import ImageTarget
from pyrit.prompt_target.image_target import SupportedDalleVersions


@pytest.fixture
def image_target():
    return ImageTarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
    )


def test_initialization_with_required_parameters(image_target: ImageTarget):
    assert image_target.deployment_name == "test"
    assert image_target.image_target is not None


def test_initialization_invalid_num_images():
    with pytest.raises(ValueError):
        ImageTarget(
            deployment_name="test",
            endpoint="test",
            api_key="test",
            dalle_version=SupportedDalleVersions.V3,
            num_images=3,
        )


@patch("pyrit.prompt_target.image_target.ImageTarget._generate_images")
def test_send_prompt(mock_image, image_target):
    mock_image.return_value = {"Mock Image: ": "mock value"}
    resp = image_target.send_prompt(normalized_prompt="test prompt", normalizer_id="1", conversation_id="2")
    print("SYNC RESP: ", resp)
    assert resp


@patch("pyrit.prompt_target.image_target.ImageTarget._generate_images_async")
@pytest.mark.asyncio
async def test_send_prompt_async(mock_image, image_target):
    mock_image.return_value = {"Mock Image: ": "mock value"}
    resp = await image_target.send_prompt_async(normalized_prompt="test prompt", normalizer_id="1", conversation_id="2")
    assert resp
