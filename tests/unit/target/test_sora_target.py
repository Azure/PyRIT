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
from pyrit.models import Message, MessagePiece
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
        "n_seconds": 5,
        "height": 480,
        "width": 480,
        "failure_reason": None,
    }


@pytest.fixture
def sample_conversations() -> MutableSequence[MessagePiece]:
    conversations = get_sample_conversations()
    return Message.flatten_to_message_pieces(conversations)


def test_initialization_with_required_parameters(sora_target: OpenAISoraTarget):
    assert sora_target
    assert sora_target._height == "720"
    assert sora_target._width == "1280"
    assert sora_target._n_seconds == 4
    # API version is detected from endpoint URL at initialization
    assert sora_target._detected_api_version == "v1"  # Default test endpoint doesn't contain "openai/v1/videos"


def test_initialization_with_invalid_resolution_no_x(patch_central_database):
    """Test that initialization fails when resolution format doesn't contain 'x'."""
    with pytest.raises(ValueError, match="Invalid resolution format.*Expected format: 'WIDTHxHEIGHT'"):
        OpenAISoraTarget(
            endpoint="test",
            api_key="test",
            resolution_dimensions="1280720",  # Missing 'x'
        )


def test_initialization_with_invalid_resolution_multiple_x(patch_central_database):
    """Test that initialization fails when resolution format has multiple 'x' characters."""
    with pytest.raises(ValueError, match="Invalid resolution format.*Expected format: 'WIDTHxHEIGHT'"):
        OpenAISoraTarget(
            endpoint="test",
            api_key="test",
            resolution_dimensions="1280x720x480",  # Too many 'x'
        )


def test_initialization_with_invalid_resolution_empty_string(patch_central_database):
    """Test that initialization fails when resolution is an empty string."""
    with pytest.raises(ValueError, match="Invalid resolution format.*Expected format: 'WIDTHxHEIGHT'"):
        OpenAISoraTarget(
            endpoint="test",
            api_key="test",
            resolution_dimensions="",  # Empty string
        )


# NOTE: Validation tests removed - with runtime API detection, parameter validation
# is now handled by the API itself rather than at initialization time. Invalid
# parameters will result in API errors when send_prompt_async is called.


@pytest.mark.asyncio
async def test_send_prompt_async_succeeded_download(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
    video_generation_response_success: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_success)

    sora_target.check_job_status_async = AsyncMock()
    sora_target.check_job_status_async.return_value.content = json.dumps(video_generation_response_success)
    sora_target.download_video_content_async = AsyncMock()
    sora_target.download_video_content_async.return_value.status_code = 200
    sora_target.download_video_content_async.return_value.content = b"video data"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=Message([request]))
        path = response.get_value()
        assert path

        job_id = video_generation_response_success["id"]
        generation_id = video_generation_response_success["generations"][0]["id"]
        assert path.endswith(f"{job_id}_{generation_id}.mp4")
        assert os.path.exists(path)

        with open(path, "r") as file:
            data = file.read()
            assert data == "video data"

        os.remove(path)


@pytest.mark.asyncio
async def test_send_prompt_async_succeeded_download_error(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
    video_generation_response_success: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_success)

    sora_target.check_job_status_async = AsyncMock()
    sora_target.check_job_status_async.return_value.content = json.dumps(video_generation_response_success)
    sora_target.download_video_content_async = AsyncMock()
    sora_target.download_video_content_async.return_value.status_code = 400
    sora_target.download_video_content_async.return_value.content = b"error"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=Message([request]))
        response_content = response.message_pieces[0]

        response_content.original_value = f"Status Code: 400, Message: {sora_target.download_video_content_async}"
        response_content.response_error = "unknown"
        response_content.original_value_data_type = "error"


@pytest.mark.asyncio
async def test_send_prompt_async_failed_unknown(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
    video_generation_response_failure_unknown: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_failure_unknown)

    sora_target.check_job_status_async = AsyncMock()
    sora_target.check_job_status_async.return_value.content = json.dumps(video_generation_response_failure_unknown)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=Message([request]))
        response_content = response.message_pieces[0]
        response_content.original_value = "task_03 failed, Reason: other"
        response_content.response_error = "unknown"
        response_content.original_value_data_type = "error"


@pytest.mark.asyncio
async def test_send_prompt_async_failed_moderation(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
    video_generation_response_failure_moderation: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_failure_moderation)

    sora_target.check_job_status_async = AsyncMock()
    sora_target.check_job_status_async.return_value.content = json.dumps(video_generation_response_failure_moderation)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=Message([request]))
        response_content = response.message_pieces[0]
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

        generation_id = video_generation_response_success["generations"][0]["id"]
        task_id = video_generation_response_success["id"]

        # Set detected API version to v1 for this test
        sora_target._detected_api_version = "v1"

        with pytest.raises(RetryError):
            await sora_target.download_video_content_async(task_id=task_id, generation_id=generation_id)

        max_attempts = os.getenv("CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS")
        if max_attempts:
            assert mock_request.call_count == int(max_attempts)


@pytest.mark.asyncio
async def test_send_prompt_async_timeout(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
    video_generation_response: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response)

    sora_target.check_job_status_async = AsyncMock()
    sora_target.check_job_status_async.return_value.content = json.dumps(video_generation_response)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=Message([request]))
        response_content = response.message_pieces[0]

        job_id = video_generation_response["id"]
        task_status = video_generation_response["status"]
        response_content.original_value = f"{job_id} {task_status}, Response {str(video_generation_response)}"


@pytest.mark.asyncio
async def test_check_job_status_async_custom_retry(
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
        sora_target._detected_api_version = "v1"

        with pytest.raises(RetryError):
            await sora_target.check_job_status_async(task_id=task_id)

        assert mock_request.call_count == sora_target.CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test that RateLimitException is raised for 429 status."""
    request = sample_conversations[0]

    response = MagicMock()
    response.status_code = 429

    side_effect = httpx.HTTPStatusError("Rate Limit Exception", response=response, request=MagicMock())

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:

        with pytest.raises(RateLimitException):
            await sora_target.send_prompt_async(prompt_request=Message([request]))

        max_attempts = os.getenv("RETRY_MAX_NUM_ATTEMPTS")
        if max_attempts:
            assert mock_request.call_count == int(max_attempts)


@pytest.mark.asyncio
async def test_send_prompt_async_http_error_handled(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test that HTTPStatusError (400) is handled gracefully and returns an error message."""
    request = sample_conversations[0]

    response = MagicMock()
    response.status_code = 400
    response.json.return_value = {"error": {"message": "Bad Request"}}

    side_effect = httpx.HTTPStatusError("Bad Request", response=response, request=MagicMock())

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:

        result = await sora_target.send_prompt_async(prompt_request=Message([request]))

        # Should return an error message, not raise an exception
        assert result is not None
        assert result.message_pieces[0].response_error == "unknown"
        assert "400" in result.message_pieces[0].converted_value
        assert mock_request.call_count == 1


@pytest.mark.asyncio
async def test_check_task_exceptions(
    sora_target: OpenAISoraTarget,
):
    response = MagicMock()
    response.status_code = 429
    side_effect = httpx.HTTPStatusError("Rate Limit Exception", response=response, request=MagicMock())

    # Set detected API version to v1 for this test
    sora_target._detected_api_version = "v1"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:

        with pytest.raises(RateLimitException) as e:
            await sora_target.check_job_status_async(task_id="task_id")
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

    # Set detected API version to v1 for this test
    sora_target._detected_api_version = "v1"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:

        with pytest.raises(RateLimitException) as e:
            await sora_target.download_video_content_async(task_id="task_id", generation_id="generation_id")
            assert str(e.value) == "Status Code: 429, Message: Rate Limit Exception"

            max_attempts = os.getenv("RETRY_MAX_NUM_ATTEMPTS")
            if max_attempts:
                assert mock_request.call_count == int(max_attempts)


def test_api_version_detected_v1_from_endpoint(patch_central_database):
    """Test that v1 API is detected from endpoint URL without 'openai/v1/videos'."""
    target = OpenAISoraTarget(
        endpoint="https://example.com/deployments/sora-turbo",
        api_key="test",
    )
    assert target._detected_api_version == "v1"


def test_api_version_detected_v2_from_endpoint(patch_central_database):
    """Test that v2 API is detected from endpoint URL containing 'openai/v1/videos'."""
    target = OpenAISoraTarget(
        endpoint="https://example.com/openai/v1/videos/generations",
        api_key="test",
    )
    assert target._detected_api_version == "v2"


@pytest.mark.asyncio
async def test_send_prompt_async_uses_v1_api(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
    video_generation_response_success: dict,
):
    """Test that the target uses Sora-1 API when detected."""
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_success)

    sora_target.check_job_status_async = AsyncMock()
    sora_target.check_job_status_async.return_value.content = json.dumps(video_generation_response_success)
    sora_target.download_video_content_async = AsyncMock()
    sora_target.download_video_content_async.return_value.status_code = 200
    sora_target.download_video_content_async.return_value.content = b"video data"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=Message([request]))

        # Verify v1 API was detected
        assert sora_target._detected_api_version == "v1"

        # Verify v1 API was called with correct parameters
        call_args = mock_request.call_args
        assert call_args is not None
        assert "/jobs" in str(call_args)
        assert call_args.kwargs.get("method") == "POST"

        # Verify response is valid
        path = response.get_value()
        assert path
        job_id = video_generation_response_success["id"]
        generation_id = video_generation_response_success["generations"][0]["id"]
        assert path.endswith(f"{job_id}_{generation_id}.mp4")

        os.remove(path)


@pytest.mark.asyncio
async def test_send_prompt_async_uses_v2_api(
    patch_central_database,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test that the target uses Sora-2 API when detected from endpoint URL."""
    # Create target with v2 endpoint
    target = OpenAISoraTarget(
        endpoint="https://example.com/openai/v1/videos/generations",
        api_key="test",
    )

    request = sample_conversations[0]

    # Mock v2 response
    v2_response = {
        "object": "video.generation.task",
        "id": "task_v2_01",
        "status": "completed",
        "created_at": 1743551594,
        "finished_at": 1743551606,
        "generation": {
            "object": "video.generation",
            "id": "gen_v2_01",
            "task_id": "task_v2_01",
            "created_at": 1743551625,
            "size": "1280x720",
            "seconds": 4,
            "prompt": "test_success_v2",
        },
        "prompt": "test_success_v2",
        "seconds": 4,
        "size": "1280x720",
        "failure_reason": None,
    }

    v2_mock_return = MagicMock()
    v2_mock_return.content = json.dumps(v2_response)

    target.check_job_status_async = AsyncMock()
    target.check_job_status_async.return_value.content = json.dumps(v2_response)
    target.download_video_content_async = AsyncMock()
    target.download_video_content_async.return_value.status_code = 200
    target.download_video_content_async.return_value.content = b"video data v2"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = v2_mock_return

        response = await target.send_prompt_async(prompt_request=Message([request]))

        # Verify v2 API was detected from endpoint
        assert target._detected_api_version == "v2"

        # Verify v2 API was called (not /jobs endpoint)
        call_args = mock_request.call_args
        assert call_args is not None
        assert "/jobs" not in str(call_args)
        assert call_args.kwargs.get("method") == "POST"
        assert call_args.kwargs.get("files") is not None  # v2 uses multipart form data

        # Verify response is valid
        path = response.get_value()
        assert path
        task_id = v2_response["id"]
        assert path.endswith(f"{task_id}.mp4")

        os.remove(path)


@pytest.mark.asyncio
async def test_check_job_status_async_v1(
    sora_target: OpenAISoraTarget,
    video_generation_response_success: dict,
):
    """Test check_job_status_async uses correct endpoint for v1 API."""
    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(video_generation_response_success)

    sora_target._detected_api_version = "v1"
    task_id = video_generation_response_success["id"]

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = openai_mock_return

        result = await sora_target.check_job_status_async(task_id=task_id)

        # Verify v1 endpoint was used
        call_args = mock_request.call_args
        assert call_args is not None
        assert f"/jobs/{task_id}" in str(call_args)
        assert result.content == json.dumps(video_generation_response_success)


@pytest.mark.asyncio
async def test_check_job_status_async_v2(
    sora_target: OpenAISoraTarget,
):
    """Test check_job_status_async uses correct endpoint for v2 API."""
    v2_response = {
        "object": "video.generation.task",
        "id": "task_v2_01",
        "status": "succeeded",
        "created_at": 1743551594,
        "finished_at": 1743551606,
        "generation": {
            "object": "video.generation",
            "id": "gen_v2_01",
            "task_id": "task_v2_01",
            "created_at": 1743551625,
            "size": "1280x720",
            "seconds": 4,
            "prompt": "test",
        },
        "prompt": "test",
        "seconds": 4,
        "size": "1280x720",
        "failure_reason": None,
    }

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(v2_response)

    sora_target._detected_api_version = "v2"
    task_id = v2_response["id"]

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = openai_mock_return

        result = await sora_target.check_job_status_async(task_id=task_id)

        # Verify v2 endpoint was used (no /jobs prefix)
        call_args = mock_request.call_args
        assert call_args is not None
        # v2 uses /{task_id} not /jobs/{task_id}
        assert "/jobs" not in str(call_args)
        assert result.content == json.dumps(v2_response)


@pytest.mark.asyncio
async def test_download_video_content_async_v1(
    sora_target: OpenAISoraTarget,
):
    """Test download_video_content_async uses correct endpoint for v1 API."""
    openai_mock_return = MagicMock()
    openai_mock_return.status_code = 200
    openai_mock_return.content = b"video data v1"

    sora_target._detected_api_version = "v1"
    task_id = "task_01"
    generation_id = "gen_01"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = openai_mock_return

        result = await sora_target.download_video_content_async(task_id=task_id, generation_id=generation_id)

        # Verify v1 endpoint was used
        call_args = mock_request.call_args
        assert call_args is not None
        assert f"/{generation_id}/content/video" in str(call_args)
        assert result.content == b"video data v1"


@pytest.mark.asyncio
async def test_download_video_content_async_v2(
    sora_target: OpenAISoraTarget,
):
    """Test download_video_content_async uses correct endpoint for v2 API."""
    openai_mock_return = MagicMock()
    openai_mock_return.status_code = 200
    openai_mock_return.content = b"video data v2"

    sora_target._detected_api_version = "v2"
    task_id = "task_v2_01"

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = openai_mock_return

        result = await sora_target.download_video_content_async(task_id=task_id)

        # Verify v2 endpoint was used
        call_args = mock_request.call_args
        assert call_args is not None
        assert f"/{task_id}/content" in str(call_args)
        assert result.content == b"video data v2"


@pytest.mark.asyncio
async def test_download_video_content_async_v1_missing_generation_id(
    sora_target: OpenAISoraTarget,
):
    """Test that v1 API requires generation_id."""
    sora_target._detected_api_version = "v1"

    with pytest.raises(ValueError, match="generation_id required for Sora v1 API"):
        await sora_target.download_video_content_async(task_id="test_task", generation_id=None)


@pytest.mark.asyncio
async def test_send_v2_request_content_filter_error(
    patch_central_database,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test v2 API error handling for content filter errors."""
    target = OpenAISoraTarget(
        endpoint="https://example.com/openai/v1/videos/generations",
        api_key="test",
    )

    request = sample_conversations[0]

    # Mock 400 error with content policy violation
    error_response = MagicMock()
    error_response.status_code = 400
    error_response.json.return_value = {
        "error": {"message": "Content policy violation detected", "type": "content_filter", "code": "content_filter"}
    }
    error_response.text = "Content filter error"

    http_error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=error_response)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.side_effect = http_error

        response = await target.send_prompt_async(prompt_request=Message([request]))

        # Should handle as content filter
        assert response.message_pieces[0].response_error == "blocked"


@pytest.mark.asyncio
async def test_send_v2_request_non_content_filter_error(
    patch_central_database,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test v2 API error handling for non-content-filter errors."""
    target = OpenAISoraTarget(
        endpoint="https://example.com/openai/v1/videos/generations",
        api_key="test",
    )

    request = sample_conversations[0]

    # Mock 500 error (not content filter)
    error_response = MagicMock()
    error_response.status_code = 500
    error_response.json.return_value = {"error": {"message": "Internal server error", "type": "server_error"}}

    http_error = httpx.HTTPStatusError("Server Error", request=MagicMock(), response=error_response)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.side_effect = http_error

        response = await target.send_prompt_async(prompt_request=Message([request]))

        # Should handle as unknown error
        assert response.message_pieces[0].response_error == "unknown"
        assert "500" in response.message_pieces[0].converted_value


@pytest.mark.asyncio
async def test_send_v2_request_error_non_dict(
    patch_central_database,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test v2 API error handling when error field is not a dict."""
    target = OpenAISoraTarget(
        endpoint="https://example.com/openai/v1/videos/generations",
        api_key="test",
    )

    request = sample_conversations[0]

    # Mock error with non-dict error field
    error_response = MagicMock()
    error_response.status_code = 400
    error_response.json.return_value = {"error": "Simple error string"}

    http_error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=error_response)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.side_effect = http_error

        response = await target.send_prompt_async(prompt_request=Message([request]))

        assert "Simple error string" in response.message_pieces[0].converted_value


@pytest.mark.asyncio
async def test_send_v2_request_error_unparseable_json(
    patch_central_database,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test v2 API error handling when response JSON is unparseable."""
    target = OpenAISoraTarget(
        endpoint="https://example.com/openai/v1/videos/generations",
        api_key="test",
    )

    request = sample_conversations[0]

    # Mock error with unparseable JSON
    error_response = MagicMock()
    error_response.status_code = 400
    error_response.json.side_effect = Exception("Invalid JSON")
    error_response.text = "Raw error response text"

    http_error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=error_response)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.side_effect = http_error

        response = await target.send_prompt_async(prompt_request=Message([request]))

        assert "Raw error response text" in response.message_pieces[0].converted_value


@pytest.mark.asyncio
async def test_handle_response_cancelled_status(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test handling of CANCELLED job status."""
    request = sample_conversations[0]

    cancelled_response = {
        "id": "task_cancelled",
        "status": "cancelled",
        "failure_reason": None,
    }

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(cancelled_response)

    sora_target.check_job_status_async = AsyncMock()
    sora_target.check_job_status_async.return_value.content = json.dumps(cancelled_response)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=Message([request]))

        assert response.message_pieces[0].response_error == "unknown"
        assert "cancelled" in response.message_pieces[0].converted_value.lower()


@pytest.mark.asyncio
async def test_handle_response_internal_error(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test handling of FAILED job with internal_error reason."""
    request = sample_conversations[0]

    failed_response = {
        "id": "task_internal_error",
        "status": "failed",
        "failure_reason": "internal_error",
    }

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(failed_response)

    sora_target.check_job_status_async = AsyncMock()
    sora_target.check_job_status_async.return_value.content = json.dumps(failed_response)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=Message([request]))

        # internal_error is not a moderation error, so it's "unknown"
        assert response.message_pieces[0].response_error == "unknown"
        assert "internal_error" in response.message_pieces[0].converted_value


@pytest.mark.asyncio
async def test_handle_response_error_dict_fields(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test handling of error response with dict containing message/type/code."""
    request = sample_conversations[0]

    failed_response = {
        "id": "task_error_details",
        "status": "failed",
        "failure_reason": None,
        "error": {"message": "Detailed error message", "type": "validation_error", "code": "ERR_001"},
    }

    openai_mock_return = MagicMock()
    openai_mock_return.content = json.dumps(failed_response)

    sora_target.check_job_status_async = AsyncMock()
    sora_target.check_job_status_async.return_value.content = json.dumps(failed_response)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = openai_mock_return

        response = await sora_target.send_prompt_async(prompt_request=Message([request]))

        value = response.message_pieces[0].converted_value
        assert "Detailed error message" in value
        assert "validation_error" in value
        assert "ERR_001" in value


def test_validate_request_multiple_pieces(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test validation fails with multiple request pieces."""
    # Create request with 2 pieces
    request1 = sample_conversations[0]
    request2 = sample_conversations[0]

    with pytest.raises(ValueError, match="only supports a single message piece"):
        sora_target._validate_request(prompt_request=Message([request1, request2]))


def test_validate_request_non_text_type(
    sora_target: OpenAISoraTarget,
    sample_conversations: MutableSequence[MessagePiece],
):
    """Test validation fails with non-text data type."""
    request = sample_conversations[0]
    request.converted_value_data_type = "image_path"

    with pytest.raises(ValueError, match="only supports text prompt input"):
        sora_target._validate_request(prompt_request=Message([request]))


def test_is_json_response_supported(sora_target: OpenAISoraTarget):
    """Test that JSON response is not supported."""
    assert sora_target.is_json_response_supported() is False
