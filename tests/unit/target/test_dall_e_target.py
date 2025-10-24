# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import uuid
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from unit.mocks import get_image_message_piece, get_sample_conversations

from pyrit.exceptions.exception_classes import (
    EmptyResponseException,
    RateLimitException,
)
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import OpenAIDALLETarget


@pytest.fixture
def dalle_target(patch_central_database) -> OpenAIDALLETarget:
    return OpenAIDALLETarget(
        model_name="test",
        endpoint="test",
        api_key="test",
    )


@pytest.fixture
def dalle_response_json() -> dict:
    return {
        "data": [
            {
                "b64_json": "aGVsbG8=",
            }
        ],
        "model": "dall-e-3",
    }


@pytest.fixture
def sample_conversations() -> MutableSequence[MessagePiece]:
    conversations = get_sample_conversations()
    return Message.flatten_to_message_pieces(conversations)


def test_initialization_with_required_parameters(dalle_target: OpenAIDALLETarget):
    assert dalle_target
    assert dalle_target._model_name == "test"


def test_initialization_invalid_num_images():
    with pytest.raises(ValueError):
        OpenAIDALLETarget(
            model_name="test",
            endpoint="test",
            api_key="test",
            dalle_version="dall-e-3",
            num_images=3,
        )


@pytest.mark.asyncio
async def test_send_prompt_async(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
    dalle_response_json: dict,
):
    request = sample_conversations[0]

    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(dalle_response_json)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        resp = await dalle_target.send_prompt_async(prompt_request=Message([request]))
        assert resp
        path = resp.message_pieces[0].original_value
        assert os.path.isfile(path)

        with open(path, "r") as file:
            data = file.read()
            assert data == "hello"

        os.remove(path)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
    dalle_response_json: dict,
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    dalle_response_json["data"][0]["b64_json"] = ""
    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(dalle_response_json)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return

        with pytest.raises(EmptyResponseException) as e:
            await dalle_target.send_prompt_async(prompt_request=Message([request]))
            assert str(e.value) == "Status Code: 204, Message: The chat returned an empty response."
            assert mock_request.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception(
    dalle_target: OpenAIDALLETarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    response = MagicMock()
    response.status_code = 429

    side_effect = httpx.HTTPStatusError("Rate Limit Reached", response=response, request=MagicMock())

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:

        with pytest.raises(RateLimitException) as rle:
            await dalle_target.send_prompt_async(prompt_request=Message([request]))
            assert mock_request.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")
            assert str(rle.value) == "Rate Limit Reached"


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_error(
    dalle_target: OpenAIDALLETarget, sample_conversations: MutableSequence[MessagePiece]
):
    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    response = MagicMock()
    response.status_code = 400
    response.text = "Bad Request Error"  # Ensure this does NOT contain 'content_filter'

    side_effect = httpx.HTTPStatusError("Bad Request Error", response=response, request=MagicMock())

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await dalle_target.send_prompt_async(prompt_request=Message([request]))
        assert mock_request.call_count == 1
        assert str(exc_info.value) == "Bad Request Error"


@pytest.mark.asyncio
async def test_dalle_validate_request_length(dalle_target: OpenAIDALLETarget):
    request = Message(
        message_pieces=[
            MessagePiece(role="user", conversation_id="123", original_value="test"),
            MessagePiece(role="user", conversation_id="123", original_value="test2"),
        ]
    )

    with pytest.raises(ValueError, match="This target only supports a single message piece."):
        await dalle_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_dalle_validate_prompt_type(dalle_target: OpenAIDALLETarget):
    request = Message(message_pieces=[get_image_message_piece()])
    with pytest.raises(ValueError, match="This target only supports text prompt input."):
        await dalle_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_adds_memory(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
    dalle_response_json: dict,
) -> None:

    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    dalle_response_json["data"][0]["b64_json"] = ""
    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(dalle_response_json)

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:

        mock_request.return_value = openai_mock_return
        dalle_target._memory = mock_memory

        with pytest.raises(EmptyResponseException):
            await dalle_target.send_prompt_async(prompt_request=Message([request]))
            assert mock_memory.add_message_to_memory.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_adds_memory(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
) -> None:
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    response = MagicMock()
    response.status_code = 429

    side_effect = httpx.HTTPStatusError("Rate Limit Reached", response=response, request=MagicMock())

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect):

        dalle_target._memory = mock_memory

        with pytest.raises(RateLimitException):
            await dalle_target.send_prompt_async(prompt_request=Message([request]))
            assert mock_memory.add_message_to_memory.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_content_filter(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
) -> None:

    request = sample_conversations[0]
    request.conversation_id = str(uuid.uuid4())

    response = MagicMock()
    response.status_code = 400
    response.text = '{"error": {"code": "content_filter", "message": "Content filtered"}}'

    side_effect = httpx.HTTPStatusError("Bad Request Error", response=response, request=MagicMock())

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect):
        result = await dalle_target.send_prompt_async(prompt_request=Message([request]))
        # For content filter errors, a response should be returned, not an exception raised
        assert result is not None
        assert isinstance(result, Message)
        # Check that the response indicates a content filter error
        assert result.message_pieces[0].response_error == "blocked"
        assert result.message_pieces[0].converted_value_data_type == "error"


def test_is_json_response_supported(patch_central_database):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    mock_dalle_target = OpenAIDALLETarget(model_name="test", endpoint="test", api_key="test")
    assert mock_dalle_target.is_json_response_supported() is False


@pytest.mark.asyncio
async def test_dalle_target_no_api_version(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
    dalle_response_json: dict,
):
    target = OpenAIDALLETarget(
        api_key="test_key", endpoint="https://mock.azure.com", model_name="dalle-3", api_version=None
    )
    request = Message([sample_conversations[0]])

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = MagicMock()
        mock_request.return_value.status_code = 200
        mock_request.return_value.text = json.dumps(dalle_response_json)

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" not in called_params


@pytest.mark.asyncio
async def test_dalle_target_default_api_version(
    dalle_target: OpenAIDALLETarget,
    sample_conversations: MutableSequence[MessagePiece],
    dalle_response_json: dict,
):
    target = OpenAIDALLETarget(api_key="test_key", endpoint="https://mock.azure.com", model_name="dalle-3")
    request = Message([sample_conversations[0]])

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = MagicMock()
        mock_request.return_value.status_code = 200
        mock_request.return_value.text = json.dumps(dalle_response_json)

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" in called_params
        assert called_params["api-version"] == "2024-10-21"


@pytest.mark.asyncio
async def test_send_prompt_async_calls_refresh_auth_headers(dalle_target):
    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = []
    mock_memory.add_message_to_memory = AsyncMock()

    dalle_target._memory = mock_memory

    dalle_target.refresh_auth_headers = MagicMock()
    dalle_target._validate_request = MagicMock()
    dalle_target._construct_request_body = AsyncMock(return_value={})

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async") as mock_make_request:
        mock_response = MagicMock()
        mock_response.text = json.dumps({"data": [{"b64_json": "aGVsbG8="}]})  # Base64 encoded "hello"
        mock_make_request.return_value = mock_response

        prompt_request = Message(
            message_pieces=[
                MessagePiece(
                    role="user",
                    original_value="test prompt",
                    converted_value="test prompt",
                    converted_value_data_type="text",
                )
            ]
        )
        await dalle_target.send_prompt_async(prompt_request=prompt_request)

        dalle_target.refresh_auth_headers.assert_called_once()


# URL Validation Tests
def test_dalle_target_url_validation_valid_azure_endpoint_no_warning(caplog, patch_central_database):
    """Test that valid Azure DALL-E endpoint doesn't trigger warning."""
    valid_endpoint = "https://myservice.openai.azure.com/openai/deployments/dall-e-3/images/generations"

    with patch.dict(os.environ, {}, clear=True):
        with caplog.at_level(logging.WARNING):
            target = OpenAIDALLETarget(
                model_name="dall-e-3", endpoint=valid_endpoint, api_key="test-key", api_version="2024-10-21"
            )

    # Should not have any warnings
    warning_logs = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warning_logs) == 0
    assert target


def test_dalle_target_url_validation_invalid_endpoint_triggers_warning(caplog, patch_central_database):
    """Test that invalid DALL-E endpoint triggers warning."""
    invalid_endpoint = "https://api.openai.com/v1/wrong/path"

    with patch.dict(os.environ, {}, clear=True):
        with caplog.at_level(logging.WARNING):
            target = OpenAIDALLETarget(
                model_name="dall-e-3", endpoint=invalid_endpoint, api_key="test-key", api_version="2024-10-21"
            )

    # Should have a warning
    warning_logs = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warning_logs) >= 1
    endpoint_warnings = [log for log in warning_logs if "Please verify your endpoint" in log.message]
    assert len(endpoint_warnings) == 1
    assert target
    assert "/openai/deployments/*/images/generations" in endpoint_warnings[0].message


def test_dalle_target_url_validation_wildcard_pattern_matching(caplog, patch_central_database):
    """Test wildcard pattern matching with various Azure deployment names."""
    test_cases = [
        ("https://service.openai.azure.com/openai/deployments/dall-e-3/images/generations", True),
        ("https://service.openai.azure.com/openai/deployments/my-custom-dalle/images/generations", True),
        ("https://service.openai.azure.com/openai/deployments/dall-e-3/wrong/generations", False),
    ]

    for endpoint, should_be_valid in test_cases:
        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                caplog.clear()  # Clear previous logs
                target = OpenAIDALLETarget(
                    model_name="dall-e-3", endpoint=endpoint, api_key="test-key", api_version="2024-10-21"
                )

            warning_logs = [record for record in caplog.records if record.levelno >= logging.WARNING]

            if should_be_valid:
                assert len(warning_logs) == 0, f"Expected no warning for {endpoint}"
                assert target is not None, f"Target should be created for {endpoint}"
            else:
                endpoint_warnings = [log for log in warning_logs if "Please verify your endpoint" in log.message]
                assert len(endpoint_warnings) >= 1, f"Expected warning for {endpoint}"
                assert target is not None, f"Target should be created even with warning for {endpoint}"
