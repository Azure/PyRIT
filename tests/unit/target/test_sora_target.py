# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_sample_conversations

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import OpenAISoraTarget


@pytest.fixture
def sora_target(patch_central_database) -> OpenAISoraTarget:
    return OpenAISoraTarget(
        endpoint="test",
        api_key="test",
    )


@pytest.fixture
def video_generation_response_success() -> dict:
    return {
        "object": "video.generation.job",
        "id": "task_01",
        "status": "succeeded",
        "created_at": 1743551594,
        "finished_at": 1743551606,
        "generations": [
            {
                "object": "video.generation",
                "id": "gen_01",
                "job_id": "task_01",
                "created_at": 1743551625,
                "width": 480,
                "height": 480,
                "n_seconds": 5,
                "prompt": "test",
            }
        ],
        "prompt": "test",
        "n_variants": 1,
        "n_seconds": 5,
        "height": 480,
        "width": 480,
        "failure_reason": None,
    }


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


def test_initialization_with_required_parameters(sora_target: OpenAISoraTarget):
    assert sora_target
    assert sora_target._height == "480"
    assert sora_target._width == "480"
    assert sora_target._n_seconds == 5
    assert sora_target._n_variants == 1
    assert sora_target._output_filename == None
    assert sora_target._api_version == "2025-02-15-preview"


@pytest.mark.parametrize(
    "resolution_dimensions, n_seconds, n_variants, err_msg",
    [
        (
            "1080x1080",
            15,
            1,
            "n_seconds must be less than or equal to 10 for resolution dimensions of 1080x1080 or 1920x1080.",
        ),
        (
            "1920x1080",
            5,
            2,
            "n_variants must be less than or equal to 1 for resolution dimensions of 1080x1080 or 1920x1080.",
        ),
        (
            "720x720",
            25,
            1,
            "n_seconds must be less than or equal to 20 for resolution dimensions other than 1080x1080 or 1920x1080.",
        ),
        (
            "1280x720",
            5,
            3,
            "n_variants must be less than or equal to 2 for resolution dimensions of 720x720 or 1280x720.",
        ),
        (
            "480x480",
            25,
            1,
            "n_seconds must be less than or equal to 20 for resolution dimensions other than 1080x1080 or 1920x1080.",
        ),
    ],
)
def test_initialization_invalid_input(resolution_dimensions, n_seconds, n_variants, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        OpenAISoraTarget(
            endpoint="test",
            api_key="test",
            resolution_dimensions=resolution_dimensions,
            n_seconds=n_seconds,
            n_variants=n_variants,
        )


@pytest.mark.asyncio
async def test_send_prompt_file_save_async(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    video_generation_response_success: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(video_generation_response_success)

    sora_target.check_task_status = AsyncMock()
    sora_target.check_task_status.return_value = json.dumps(video_generation_response_success)
    sora_target.download_video_content = AsyncMock()
    sora_target.download_video_content.return_value.status_code = 200
    sora_target.download_video_content.return_value.content = b"video data"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
        path = response.get_value()
        assert path
        assert path.endswith(".mp4")
        assert os.path.exists(path)

        with open(path, "r") as file:
            data = file.read()
            assert data == "video data"

        os.remove(path)
