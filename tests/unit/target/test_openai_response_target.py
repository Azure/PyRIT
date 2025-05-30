# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from tempfile import NamedTemporaryFile
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import BadRequestError, RateLimitError
from unit.mocks import (
    get_image_request_piece,
    get_sample_conversations,
    openai_response_json_dict,
)

from pyrit.exceptions.exception_classes import (
    EmptyResponseException,
    PyritException,
    RateLimitException,
)
from pyrit.memory.duckdb_memory import DuckDBMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import OpenAIResponseTarget


def fake_construct_response_from_request(request, response_text_pieces):
    return {"dummy": True, "request": request, "response": response_text_pieces}


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def dummy_text_request_piece() -> PromptRequestPiece:
    return PromptRequestPiece(
        role="user",
        conversation_id="dummy_convo",
        original_value="dummy text",
        converted_value="dummy text",
        original_value_data_type="text",
        converted_value_data_type="text",
    )


@pytest.fixture
def target(patch_central_database) -> OpenAIResponseTarget:
    return OpenAIResponseTarget(
        model_name="gpt-o",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        api_version="some_version",
    )


@pytest.fixture
def openai_response_json() -> dict:
    return openai_response_json_dict()


def test_init_with_no_deployment_var_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            OpenAIResponseTarget()


def test_init_with_no_endpoint_uri_var_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            OpenAIResponseTarget(
                model_name="gpt-4",
                endpoint="",
                api_key="xxxxx",
                api_version="some_version",
            )


def test_init_with_no_additional_request_headers_var_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            OpenAIResponseTarget(
                model_name="gpt-4", endpoint="", api_key="xxxxx", api_version="some_version", headers=""
            )


@pytest.mark.asyncio()
async def test_build_input_for_multi_modal(target: OpenAIResponseTarget):

    image_request = get_image_request_piece()
    entries = [
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    original_value_data_type="text",
                    original_value="Hello 1",
                ),
                image_request,
            ]
        ),
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value_data_type="text",
                    original_value="Hello 2",
                ),
            ]
        ),
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    original_value_data_type="text",
                    original_value="Hello 3",
                ),
                image_request,
            ]
        ),
    ]
    with patch(
        "pyrit.common.data_url_converter.convert_local_image_to_data_url",
        return_value="data:image/jpeg;base64,encoded_string",
    ):
        messages = await target._build_input_for_multi_modal_async(entries)

    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "input_text"  # type: ignore
    assert messages[0]["content"][1]["type"] == "input_image"  # type: ignore
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["type"] == "output_text"  # type: ignore
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "input_text"  # type: ignore
    assert messages[2]["content"][1]["type"] == "input_image"  # type: ignore

    os.remove(image_request.original_value)


@pytest.mark.asyncio
async def test_build_input_for_multi_modal_with_unsupported_data_types(target: OpenAIResponseTarget):
    # Like an image_path, the audio_path requires a file, but doesn't validate any contents
    entry = get_image_request_piece()
    entry.converted_value_data_type = "audio_path"

    with pytest.raises(ValueError) as excinfo:
        await target._build_input_for_multi_modal_async([PromptRequestResponse(request_pieces=[entry])])
    assert "Multimodal data type audio_path is not yet supported." in str(excinfo.value)


@pytest.mark.asyncio
async def test_construct_request_body_includes_extra_body_params(
    patch_central_database, dummy_text_request_piece: PromptRequestPiece
):
    target = OpenAIResponseTarget(
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        api_version="some_version",
        extra_body_parameters={"key": "value"},
    )

    request = PromptRequestResponse(request_pieces=[dummy_text_request_piece])

    body = await target._construct_request_body(conversation=[request], is_json_response=False)
    assert body["key"] == "value"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_json", [True, False])
async def test_construct_request_body_includes_json(
    is_json, target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece
):

    request = PromptRequestResponse(request_pieces=[dummy_text_request_piece])

    body = await target._construct_request_body(conversation=[request], is_json_response=is_json)
    if is_json:
        assert body["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in body


@pytest.mark.asyncio
async def test_construct_request_body_removes_empty_values(
    target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece
):
    request = PromptRequestResponse(request_pieces=[dummy_text_request_piece])

    body = await target._construct_request_body(conversation=[request], is_json_response=False)
    assert "max_completion_tokens" not in body
    assert "max_tokens" not in body
    assert "temperature" not in body
    assert "top_p" not in body
    assert "frequency_penalty" not in body
    assert "presence_penalty" not in body


@pytest.mark.asyncio
async def test_construct_request_body_serializes_text_message(
    target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece
):
    request = PromptRequestResponse(request_pieces=[dummy_text_request_piece])

    body = await target._construct_request_body(conversation=[request], is_json_response=False)
    assert body["input"][0]["content"][0]["text"] == "dummy text"


@pytest.mark.asyncio
async def test_construct_request_body_serializes_complex_message(
    target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece
):
    request = PromptRequestResponse(request_pieces=[dummy_text_request_piece, get_image_request_piece()])

    body = await target._construct_request_body(conversation=[request], is_json_response=False)
    messages = body["input"][0]["content"]
    assert len(messages) == 2
    assert messages[0]["type"] == "input_text"
    assert messages[1]["type"] == "input_image"


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_adds_to_memory(
    openai_response_json: dict, target: OpenAIResponseTarget
):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    target._memory = mock_memory

    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
    assert os.path.exists(tmp_file_name)
    prompt_req_resp = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                conversation_id="12345679",
                original_value="hello",
                converted_value="hello",
                original_value_data_type="text",
                converted_value_data_type="text",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            ),
            PromptRequestPiece(
                role="user",
                conversation_id="12345679",
                original_value=tmp_file_name,
                converted_value=tmp_file_name,
                original_value_data_type="image_path",
                converted_value_data_type="image_path",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            ),
        ]
    )
    # Make assistant response empty
    openai_response_json["output"][0]["content"][0]["text"] = ""

    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(openai_response_json)

    with patch(
        "pyrit.common.data_url_converter.convert_local_image_to_data_url",
        return_value="data:image/jpeg;base64,encoded_string",
    ):
        with patch(
            "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = openai_mock_return
            target._memory = MagicMock(MemoryInterface)

            with pytest.raises(EmptyResponseException):
                await target.send_prompt_async(prompt_request=prompt_req_resp)

            assert mock_create.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception_adds_to_memory(
    target: OpenAIResponseTarget,
):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    target._memory = mock_memory

    response = MagicMock()
    response.status_code = 429

    side_effect = httpx.HTTPStatusError("Rate Limit Reached", response=response, request=MagicMock())

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect):

        prompt_request = PromptRequestResponse(
            request_pieces=[PromptRequestPiece(role="user", conversation_id="123", original_value="Hello")]
        )

        with pytest.raises(RateLimitException) as rle:
            await target.send_prompt_async(prompt_request=prompt_request)
            target._memory.get_conversation.assert_called_once_with(conversation_id="123")
            target._memory.add_request_response_to_memory.assert_called_once_with(request=prompt_request)

            assert str(rle.value) == "Rate Limit Reached"


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_error_adds_to_memory(target: OpenAIResponseTarget):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    target._memory = mock_memory

    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="123", original_value="Hello")]
    )

    response = MagicMock()
    response.status_code = 400
    response.text = "Some error text"

    side_effect = httpx.HTTPStatusError("Bad Request", response=response, request=MagicMock())

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect):
        with pytest.raises(httpx.HTTPStatusError) as bre:
            await target.send_prompt_async(prompt_request=prompt_request)
            target._memory.get_conversation.assert_called_once_with(conversation_id="123")
            target._memory.add_request_response_to_memory.assert_called_once_with(request=prompt_request)

            assert str(bre.value) == "Bad Request"


@pytest.mark.asyncio
async def test_send_prompt_async(openai_response_json: dict, target: OpenAIResponseTarget):
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
    assert os.path.exists(tmp_file_name)
    prompt_req_resp = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                conversation_id="12345679",
                original_value="hello",
                converted_value="hello",
                original_value_data_type="text",
                converted_value_data_type="text",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            ),
            PromptRequestPiece(
                role="user",
                conversation_id="12345679",
                original_value=tmp_file_name,
                converted_value=tmp_file_name,
                original_value_data_type="image_path",
                converted_value_data_type="image_path",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            ),
        ]
    )
    with patch(
        "pyrit.common.data_url_converter.convert_local_image_to_data_url",
        return_value="data:image/jpeg;base64,encoded_string",
    ):
        with patch(
            "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
        ) as mock_create:
            openai_mock_return = MagicMock()
            openai_mock_return.text = json.dumps(openai_response_json)
            mock_create.return_value = openai_mock_return
            response: PromptRequestResponse = await target.send_prompt_async(prompt_request=prompt_req_resp)
            assert len(response.request_pieces) == 1
            assert response.get_value() == "hi"
    os.remove(tmp_file_name)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_retries(openai_response_json: dict, target: OpenAIResponseTarget):
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
    assert os.path.exists(tmp_file_name)
    prompt_req_resp = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                conversation_id="12345679",
                original_value="hello",
                converted_value="hello",
                original_value_data_type="text",
                converted_value_data_type="text",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            ),
            PromptRequestPiece(
                role="user",
                conversation_id="12345679",
                original_value=tmp_file_name,
                converted_value=tmp_file_name,
                original_value_data_type="image_path",
                converted_value_data_type="image_path",
                prompt_target_identifier={"target": "target-identifier"},
                orchestrator_identifier={"test": "test"},
                labels={"test": "test"},
            ),
        ]
    )
    # Make assistant response empty
    openai_response_json["output"][0]["content"][0]["text"] = ""
    with patch(
        "pyrit.common.data_url_converter.convert_local_image_to_data_url",
        return_value="data:image/jpeg;base64,encoded_string",
    ):
        with patch(
            "pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock
        ) as mock_create:

            openai_mock_return = MagicMock()
            openai_mock_return.text = json.dumps(openai_response_json)
            mock_create.return_value = openai_mock_return
            target._memory = MagicMock(MemoryInterface)

            with pytest.raises(EmptyResponseException):
                await target.send_prompt_async(prompt_request=prompt_req_resp)

            assert mock_create.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.asyncio
async def test_send_prompt_async_rate_limit_exception_retries(target: OpenAIResponseTarget):

    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="12345", original_value="Hello")]
    )

    response = MagicMock()
    response.status_code = 429

    side_effect = RateLimitError("Rate Limit Reached", response=response, body="Rate limit reached")

    with patch(
        "pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect
    ) as mock_request:

        with pytest.raises(RateLimitError):
            await target.send_prompt_async(prompt_request=prompt_request)
            assert mock_request.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_send_prompt_async_bad_request_error(target: OpenAIResponseTarget):

    response = MagicMock()
    response.status_code = 400

    side_effect = BadRequestError("Bad Request Error", response=response, body="Bad request")

    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="1236748", original_value="Hello")]
    )

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect):
        with pytest.raises(BadRequestError) as bre:
            await target.send_prompt_async(prompt_request=prompt_request)
            assert str(bre.value) == "Bad Request Error"


@pytest.mark.asyncio
async def test_send_prompt_async_content_filter(target: OpenAIResponseTarget):

    response_body = json.dumps(
        {
            "error": {
                "code": "content_filter",
                "innererror": {
                    "code": "ResponsibleAIPolicyViolation",
                    "content_filter_result": {"violence": {"filtered": True, "severity": "medium"}},
                },
            }
        }
    )

    prompt_request = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                conversation_id="567567",
                original_value="A prompt for something harmful that gets filtered.",
            )
        ]
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = response_body

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", return_value=mock_response):
        response = await target.send_prompt_async(prompt_request=prompt_request)
        assert len(response.request_pieces) == 1
        assert response.request_pieces[0].response_error == "blocked"
        assert response.request_pieces[0].converted_value_data_type == "error"
        assert "content_filter_result" in response.get_value()


def test_validate_request_unsupported_data_types(target: OpenAIResponseTarget):

    image_piece = get_image_request_piece()
    image_piece.converted_value_data_type = "new_unknown_type"  # type: ignore
    prompt_request = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="user", original_value="Hello", converted_value_data_type="text"),
            image_piece,
        ]
    )

    with pytest.raises(ValueError) as excinfo:
        target._validate_request(prompt_request=prompt_request)

    assert "This target only supports text and image_path." in str(
        excinfo.value
    ), "Error not raised for unsupported data types"

    os.remove(image_piece.original_value)


def test_is_json_response_supported(target: OpenAIResponseTarget):
    assert target.is_json_response_supported() is True


def test_is_response_format_json_supported(target: OpenAIResponseTarget):

    request_piece = PromptRequestPiece(
        role="user",
        original_value="original prompt text",
        converted_value="Hello, how are you?",
        conversation_id="conversation_1",
        sequence=0,
        prompt_metadata={"response_format": "json"},
    )

    result = target.is_response_format_json(request_piece)

    assert result is True


def test_is_response_format_json_no_metadata(target: OpenAIResponseTarget):
    request_piece = PromptRequestPiece(
        role="user",
        original_value="original prompt text",
        converted_value="Hello, how are you?",
        conversation_id="conversation_1",
        sequence=0,
        prompt_metadata=None,
    )

    result = target.is_response_format_json(request_piece)

    assert result is False


@pytest.mark.parametrize(
    "status", ["failed", "in_progress", "cancelled", "queued", "incomplete", "some_unexpected_status"]
)
def test_construct_prompt_response_not_completed_status(
    status: str, target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece
):
    response_dict = {"status": f"{status}", "error": {"code": "some_error_code", "message": "An error occurred"}}
    response_str = json.dumps(response_dict)

    with pytest.raises(PyritException) as excinfo:
        target._construct_prompt_response_from_openai_json(
            open_ai_str_response=response_str, request_piece=dummy_text_request_piece
        )
    assert f"Message: Status {status} and error {json.dumps(response_dict["error"]).replace("\"", "'")}" in str(excinfo.value)


def test_construct_prompt_response_empty_response(
    target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece, openai_response_json
):
    openai_response_json["output"][0]["content"][0]["text"] = ""  # Simulate empty response
    response_str = json.dumps(openai_response_json)

    with pytest.raises(EmptyResponseException) as excinfo:
        target._construct_prompt_response_from_openai_json(
            open_ai_str_response=response_str, request_piece=dummy_text_request_piece
        )
    assert "The chat returned an empty response." in str(excinfo.value)


@pytest.mark.asyncio
async def test_openai_response_target_no_api_version(
    sample_conversations: MutableSequence[PromptRequestPiece], openai_response_json: dict
):
    target = OpenAIResponseTarget(
        api_key="test_key", endpoint="https://mock.azure.com", model_name="gpt-35-turbo", api_version=None
    )
    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = MagicMock()
        mock_request.return_value.status_code = 200
        mock_request.return_value.text = json.dumps(openai_response_json)

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" not in called_params


@pytest.mark.asyncio
async def test_openai_response_target_default_api_version(
    sample_conversations: MutableSequence[PromptRequestPiece], openai_response_json: dict
):
    target = OpenAIResponseTarget(api_key="test_key", endpoint="https://mock.azure.com", model_name="gpt-35-turbo")
    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = MagicMock()
        mock_request.return_value.status_code = 200
        mock_request.return_value.text = json.dumps(openai_response_json)

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" in called_params
        assert called_params["api-version"] == "2025-03-01-preview"


@pytest.mark.asyncio
async def test_send_prompt_async_calls_refresh_auth_headers(target: OpenAIResponseTarget, openai_response_json: dict):
    mock_memory = MagicMock(spec=MemoryInterface)
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    target._azure_auth = MagicMock()
    target._memory = mock_memory

    with (
        patch.object(target, "refresh_auth_headers") as mock_refresh,
        patch.object(target, "_validate_request"),
        patch.object(target, "_construct_request_body", new_callable=AsyncMock) as mock_construct,
    ):

        mock_construct.return_value = {}

        with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async") as mock_make_request:
            mock_make_request.return_value = MagicMock(text=json.dumps(openai_response_json))

            prompt_request = PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        original_value="test prompt",
                        converted_value="test prompt",
                        converted_value_data_type="text",
                    )
                ]
            )
            await target.send_prompt_async(prompt_request=prompt_request)
            mock_refresh.assert_called_once()


def test_construct_prompt_response_from_openai_json_invalid_json(
    target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece
):
    # Should raise PyritException for invalid JSON
    with pytest.raises(PyritException) as excinfo:
        target._construct_prompt_response_from_openai_json(
            open_ai_str_response="{invalid_json", request_piece=dummy_text_request_piece
        )
    assert "Failed to parse JSON response" in str(excinfo.value)


def test_construct_prompt_response_from_openai_json_no_status(
    target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece
):
    # Should raise PyritException for missing status and no content_filter error
    bad_json = json.dumps({"output": [{"type": "message", "content": [{"text": "hi"}]}]})
    with pytest.raises(PyritException) as excinfo:
        target._construct_prompt_response_from_openai_json(
            open_ai_str_response=bad_json, request_piece=dummy_text_request_piece
        )
    assert "Unexpected response format" in str(excinfo.value)


def test_construct_prompt_response_from_openai_json_reasoning(
    target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece
):
    # Should handle reasoning type and skip empty summaries
    reasoning_json = {
        "status": "completed",
        "output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "Reasoning summary."}]}],
    }
    response = target._construct_prompt_response_from_openai_json(
        open_ai_str_response=json.dumps(reasoning_json), request_piece=dummy_text_request_piece
    )
    assert response.request_pieces[0].original_value == "Reasoning summary."
    assert response.request_pieces[0].original_value_data_type == "reasoning"


def test_construct_prompt_response_from_openai_json_unsupported_type(
    target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece
):
    # Should raise ValueError for unsupported response type
    bad_type_json = {
        "status": "completed",
        "output": [{"type": "function_call", "content": [{"text": "some function call"}]}],
    }
    with pytest.raises(ValueError) as excinfo:
        target._construct_prompt_response_from_openai_json(
            open_ai_str_response=json.dumps(bad_type_json), request_piece=dummy_text_request_piece
        )
    assert "Unsupported response type" in str(excinfo.value)


def test_validate_request_allows_text_and_image(target: OpenAIResponseTarget):
    # Should not raise for valid types
    req = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="user", original_value_data_type="text", original_value="Hello"),
            PromptRequestPiece(role="user", original_value_data_type="image_path", original_value="fake.jpg"),
        ]
    )
    target._validate_request(prompt_request=req)


def test_validate_request_raises_for_invalid_type(target: OpenAIResponseTarget):
    req = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="user", original_value_data_type="audio_path", original_value="fake.mp3"),
        ]
    )
    with pytest.raises(ValueError) as excinfo:
        target._validate_request(prompt_request=req)
    assert "only supports text and image_path" in str(excinfo.value)


def test_is_json_response_supported_returns_true(target: OpenAIResponseTarget):
    assert target.is_json_response_supported() is True


@pytest.mark.asyncio
async def test_build_input_for_multi_modal_async_empty_conversation(target: OpenAIResponseTarget):
    # Should raise ValueError if no request pieces
    req = PromptRequestResponse(request_pieces=[])
    with pytest.raises(ValueError) as excinfo:
        await target._build_input_for_multi_modal_async([req])
    assert "No prompt request pieces found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_build_input_for_multi_modal_async_image_and_text(target: OpenAIResponseTarget):
    # Should build correct structure for text and image
    text_piece = PromptRequestPiece(role="user", original_value_data_type="text", original_value="hello")
    image_piece = PromptRequestPiece(role="user", original_value_data_type="image_path", original_value="fake.jpg")
    req = PromptRequestResponse(request_pieces=[text_piece, image_piece])
    with patch("pyrit.prompt_target.openai.openai_response_target.convert_local_image_to_data_url", return_value="data:image/jpeg;base64,abc"):
        result = await target._build_input_for_multi_modal_async([req])
    assert result[0]["role"] == "user"
    assert result[0]["content"][0]["type"] == "input_text"
    assert result[0]["content"][1]["type"] == "input_image"
    assert result[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_construct_request_body_filters_none(
    target: OpenAIResponseTarget, dummy_text_request_piece: PromptRequestPiece
):
    req = PromptRequestResponse(request_pieces=[dummy_text_request_piece])
    body = await target._construct_request_body([req], is_json_response=False)
    assert "max_output_tokens" not in body or body["max_output_tokens"] is None
    assert "temperature" not in body or body["temperature"] is None
    assert "top_p" not in body or body["top_p"] is None


def test_set_openai_env_configuration_vars_sets_vars():
    target = OpenAIResponseTarget(model_name="gpt", endpoint="http://test", api_key="key")
    target._set_openai_env_configuration_vars()
    assert target.model_name_environment_variable == "OPENAI_RESPONSES_MODEL"
    assert target.endpoint_environment_variable == "OPENAI_RESPONSES_ENDPOINT"
    assert target.api_key_environment_variable == "OPENAI_RESPONSES_KEY"


@pytest.mark.asyncio
async def test_build_input_for_multi_modal_async_filters_reasoning(target: OpenAIResponseTarget):
    # Prepare a conversation with a reasoning piece and a text piece
    user_prompt = PromptRequestPiece(
        role="user",
        original_value="Hello",
        converted_value="Hello",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    response_reasoning_piece = PromptRequestPiece(
        role="assistant",
        original_value="Reasoning summary.",
        converted_value="Reasoning summary.",
        original_value_data_type="reasoning",
        converted_value_data_type="reasoning",
    )
    response_text_piece = PromptRequestPiece(
        role="assistant",
        original_value="hello there",
        converted_value="hello there",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    user_followup_prompt = PromptRequestPiece(
        role="user",
        original_value="Hello indeed",
        converted_value="Hello indeed",
        original_value_data_type="text",
        converted_value_data_type="text",
    )
    conversation = [
        PromptRequestResponse(request_pieces=[user_prompt]),
        PromptRequestResponse(request_pieces=[response_reasoning_piece, response_text_piece]),
        PromptRequestResponse(request_pieces=[user_followup_prompt]),
    ]

    # Patch image conversion (should not be called)
    with patch("pyrit.common.data_url_converter.convert_local_image_to_data_url", new_callable=AsyncMock):
        result = await target._build_input_for_multi_modal_async(conversation)

    # Only the text piece should be present, reasoning should be filtered out
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[0]["content"][0]["type"] == "input_text"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"][0]["type"] == "output_text"
    assert result[2]["role"] == "user"
    assert result[2]["content"][0]["type"] == "input_text"
