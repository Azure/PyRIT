# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from tenacity import RetryError
from unit.mocks import get_sample_conversations

from pyrit.exceptions.exception_classes import RateLimitException
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
                "prompt": "test_success",
            }
        ],
        "prompt": "test_success",
        "n_variants": 1,
        "n_seconds": 5,
        "height": 480,
        "width": 480,
        "failure_reason": None,
    }


@pytest.fixture
def video_generation_response_failure_moderation() -> dict:
    return {
        "object": "video.generation.job",
        "id": "task_02",
        "status": "failed",
        "created_at": 1743556719,
        "finished_at": 1743556734,
        "generations": [
            {
                "object": "video.generation",
                "id": "gen_02",
                "job_id": "task_02",
                "created_at": 1743556759,
                "width": 480,
                "height": 480,
                "n_seconds": 5,
                "prompt": "test_failure",
            }
        ],
        "prompt": "test_failure",
        "n_variants": 1,
        "n_seconds": 5,
        "height": 480,
        "width": 480,
        "failure_reason": "output_moderation",
    }


@pytest.fixture
def video_generation_response_failure_unknown() -> dict:
    return {
        "object": "video.generation.job",
        "id": "task_03",
        "status": "failed",
        "created_at": 1743556719,
        "finished_at": 1743556734,
        "generations": [
            {
                "object": "video.generation",
                "id": "gen_03",
                "job_id": "task_03",
                "created_at": 1743556759,
                "width": 480,
                "height": 480,
                "n_seconds": 5,
                "prompt": "test_failure",
            }
        ],
        "prompt": "test_failure",
        "n_variants": 1,
        "n_seconds": 5,
        "height": 480,
        "width": 480,
        "failure_reason": "other",
    }


@pytest.fixture
def video_generation_response() -> dict:
    return {
        "object": "video.generation.job",
        "id": "task_04",
        "status": "processing",
        "created_at": 1743551594,
        "finished_at": None,
        "generations": [],
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
    assert sora_target._output_filename is None
    assert sora_target._api_version == "preview"


@pytest.mark.parametrize(
    "resolution_dimensions, n_seconds, n_variants, err_msg",
    [
        (
            "1080x1080",
            15,
            1,
            "n_seconds must be less than or equal to 10 for resolution dimensions of 1080x1080.",
        ),
        (
            "1920x1080",
            5,
            2,
            "n_variants must be less than or equal to 1 for resolution dimensions of 1920x1080.",
        ),
        (
            "720x720",
            25,
            1,
            "n_seconds must be less than or equal to 20 for resolution dimensions of 720x720.",
        ),
        (
            "1280x720",
            5,
            3,
            "n_variants must be less than or equal to 2 for resolution dimensions of 1280x720.",
        ),
        (
            "480x480",
            25,
            1,
            "n_seconds must be less than or equal to 20 for resolution dimensions of 480x480.",
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
async def test_send_prompt_async_succeeded_download(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    video_generation_response_success: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_success)

    sora_target.check_task_status_async = AsyncMock()
    sora_target.check_task_status_async.return_value.content = json.dumps(video_generation_response_success)
    sora_target.download_video_content_async = AsyncMock()
    sora_target.download_video_content_async.return_value.status_code = 200
    sora_target.download_video_content_async.return_value.content = b"video data"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
        path = response.get_value()
        assert path

        task_id = video_generation_response_success["id"]
        gen_id = video_generation_response_success["generations"][0]["id"]
        assert path.endswith(f"{task_id}_{gen_id}.mp4")
        assert os.path.exists(path)

        with open(path, "r") as file:
            data = file.read()
            assert data == "video data"

        os.remove(path)


@pytest.mark.asyncio
async def test_send_prompt_async_succeeded_download_error(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    video_generation_response_success: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_success)

    sora_target.check_task_status_async = AsyncMock()
    sora_target.check_task_status_async.return_value.content = json.dumps(video_generation_response_success)
    sora_target.download_video_content_async = AsyncMock()
    sora_target.download_video_content_async.return_value.status_code = 400
    sora_target.download_video_content_async.return_value.content = b"error"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
        response_content = response.request_pieces[0]

        response_content.original_value = f"Status Code: 400, Message: {sora_target.download_video_content_async}"
        response_content.response_error = "unknown"
        response_content.original_value_data_type = "error"


@pytest.mark.asyncio
async def test_send_prompt_async_failed_unknown(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    video_generation_response_failure_unknown: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_failure_unknown)

    sora_target.check_task_status_async = AsyncMock()
    sora_target.check_task_status_async.return_value.content = json.dumps(video_generation_response_failure_unknown)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
        response_content = response.request_pieces[0]
        response_content.original_value = "task_03 failed, Reason: other"
        response_content.response_error = "unknown"
        response_content.original_value_data_type = "error"


@pytest.mark.asyncio
async def test_send_prompt_async_failed_moderation(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    video_generation_response_failure_moderation: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_failure_moderation)

    sora_target.check_task_status_async = AsyncMock()
    sora_target.check_task_status_async.return_value.content = json.dumps(video_generation_response_failure_moderation)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
        response_content = response.request_pieces[0]
        response_content.original_value = "Status Code: 400, Message: task_02 failed, Reason: output_moderation"
        response_content.response_error = "blocked"
        response_content.original_value_data_type = "error"


@pytest.mark.asyncio
async def test_download_video_content_async_custom_retry(
    sora_target: OpenAISoraTarget,
    video_generation_response_success: dict,
):
    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_success)
    openai_mock_return.status_code = 500

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        gen_id = video_generation_response_success["generations"][0]["id"]

        with pytest.raises(RetryError):
            await sora_target.download_video_content_async(gen_id=gen_id)

        max_attempts = os.getenv("CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS")
        if max_attempts:
            assert mock_request.call_count == int(max_attempts)


@pytest.mark.asyncio
async def test_send_prompt_async_timeout(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    video_generation_response: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response)

    sora_target.check_task_status_async = AsyncMock()
    sora_target.check_task_status_async.return_value.content = json.dumps(video_generation_response)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
        response_content = response.request_pieces[0]

        task_id = video_generation_response["id"]
        task_status = video_generation_response["status"]
        response_content.original_value = f"{task_id} {task_status}, Response {str(video_generation_response)}"


@pytest.mark.asyncio
async def test_check_task_status_async_custom_retry(
    sora_target: OpenAISoraTarget,
    video_generation_response: dict,
):
    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        task_id = video_generation_response["id"]

        with pytest.raises(RetryError):
            await sora_target.check_task_status_async(task_id=task_id)

        assert mock_request.call_count == sora_target.CHECK_TASK_RETRY_MAX_NUM_ATTEMPTS


@pytest.mark.parametrize(
    "err_class, status_code, message, err_msg",
    [
        (RateLimitException, 429, "Rate Limit Exception", "Status Code: 429, Message: Rate Limit Exception"),
        (httpx.HTTPStatusError, 400, "Bad Request", "Status Code: 400, Message: Bad Request"),
    ],
)
@pytest.mark.asyncio
async def test_send_prompt_async_exceptions(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[PromptRequestPiece],
    err_class: Exception,
    status_code: int,
    message: str,
    err_msg: str,
):
    request = sample_conversations[0]

    response = MagicMock()
    response.status_code = status_code

    side_effect = httpx.HTTPStatusError(message, response=response, request=MagicMock())

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:

        with pytest.raises(err_class) as e:  # type: ignore
            await sora_target.send_prompt_async(prompt_request=PromptRequestResponse([request]))
            assert str(e.value) == err_msg

            max_attempts = os.getenv("RETRY_MAX_NUM_ATTEMPTS")
            if max_attempts and err_class == RateLimitException:
                assert mock_request.call_count == int(max_attempts)
            elif err_class == httpx.HTTPStatusError:
                assert mock_request.call_count == 1


@pytest.mark.asyncio
async def test_check_task_exceptions(
    sora_target: OpenAISoraTarget,
):
    response = MagicMock()
    response.status_code = 429
    side_effect = httpx.HTTPStatusError("Rate Limit Exception", response=response, request=MagicMock())

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:

        with pytest.raises(RateLimitException) as e:
            await sora_target.check_task_status_async(task_id="task_id")
            assert str(e.value) == "Status Code: 429, Message: Rate Limit Exception"

            max_attempts = os.getenv("RETRY_MAX_NUM_ATTEMPTS")
            if max_attempts:
                assert mock_request.call_count == int(max_attempts)


@pytest.mark.asyncio
async def test_download_video_content_async_exceptions(
    sora_target: OpenAISoraTarget,
):
    response = MagicMock()
    response.status_code = 429
    side_effect = httpx.HTTPStatusError("Rate Limit Exception", response=response, request=MagicMock())

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:

        with pytest.raises(RateLimitException) as e:
            await sora_target.download_video_content_async(gen_id="gen_id")
            assert str(e.value) == "Status Code: 429, Message: Rate Limit Exception"

            max_attempts = os.getenv("RETRY_MAX_NUM_ATTEMPTS")
            if max_attempts:
                assert mock_request.call_count == int(max_attempts)
