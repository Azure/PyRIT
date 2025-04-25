# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from contextlib import AbstractAsyncContextManager
from tempfile import NamedTemporaryFile
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
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
from pyrit.prompt_target import OpenAIChatTarget


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
def target(patch_central_database) -> OpenAIChatTarget:
    return OpenAIChatTarget(
        model_name="gpt-o",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        api_version="some_version",
    )


@pytest.fixture
def openai_response_json() -> dict:
    return openai_response_json_dict()


class MockChatCompletionsAsync(AbstractAsyncContextManager):
    async def __call__(self, *args, **kwargs):
        self.mock_chat_completion = ChatCompletion(
            id="12345678-1a2b-3c4e5f-a123-12345678abcd",
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hi"),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            created=1629389505,
            model="gpt-4",
        )
        return self.mock_chat_completion

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        pass


def test_init_with_no_deployment_var_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            OpenAIChatTarget()


def test_init_with_no_endpoint_uri_var_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            OpenAIChatTarget(
                model_name="gpt-4",
                endpoint="",
                api_key="xxxxx",
                api_version="some_version",
            )


def test_init_with_no_additional_request_headers_var_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            OpenAIChatTarget(model_name="gpt-4", endpoint="", api_key="xxxxx", api_version="some_version", headers="")


@pytest.mark.asyncio()
async def test_convert_image_to_data_url_file_not_found(target: OpenAIChatTarget):
    with pytest.raises(FileNotFoundError):
        await target._convert_local_image_to_data_url("nonexistent.jpg")


@pytest.mark.asyncio()
async def test_convert_image_with_unsupported_extension(target: OpenAIChatTarget):

    with NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name

    assert os.path.exists(tmp_file_name)

    with pytest.raises(ValueError) as exc_info:
        await target._convert_local_image_to_data_url(tmp_file_name)

    assert "Unsupported image format" in str(exc_info.value)

    os.remove(tmp_file_name)


@pytest.mark.asyncio()
@patch("os.path.exists", return_value=True)
@patch("mimetypes.guess_type", return_value=("image/jpg", None))
@patch("pyrit.models.data_type_serializer.ImagePathDataTypeSerializer")
@patch("pyrit.memory.CentralMemory.get_memory_instance", return_value=DuckDBMemory(db_path=":memory:"))
async def test_convert_image_to_data_url_success(
    mock_get_memory_instance, mock_serializer_class, mock_guess_type, mock_exists, target: OpenAIChatTarget
):
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
    mock_serializer_instance = MagicMock()
    mock_serializer_instance.read_data_base64 = AsyncMock(return_value="encoded_base64_string")
    mock_serializer_class.return_value = mock_serializer_instance

    assert os.path.exists(tmp_file_name)

    result = await target._convert_local_image_to_data_url(tmp_file_name)
    assert "data:image/jpeg;base64,encoded_base64_string" in result

    # Assertions for the mocks
    mock_serializer_class.assert_called_once_with(
        category="prompt-memory-entries", prompt_text=tmp_file_name, extension=".jpg"
    )
    mock_serializer_instance.read_data_base64.assert_called_once()

    os.remove(tmp_file_name)


@pytest.mark.asyncio()
async def test_build_chat_messages_for_multi_modal(target: OpenAIChatTarget):

    image_request = get_image_request_piece()
    entries = [
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    converted_value_data_type="text",
                    original_value="Hello",
                ),
                image_request,
            ]
        )
    ]
    with patch.object(
        target,
        "_convert_local_image_to_data_url",
        return_value="data:image/jpeg;base64,encoded_string",
    ):
        messages = await target._build_chat_messages_for_multi_modal_async(entries)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "text"  # type: ignore
    assert messages[0]["content"][1]["type"] == "image_url"  # type: ignore

    os.remove(image_request.original_value)


@pytest.mark.asyncio
async def test_build_chat_messages_for_multi_modal_with_unsupported_data_types(target: OpenAIChatTarget):
    # Like an image_path, the audio_path requires a file, but doesn't validate any contents
    entry = get_image_request_piece()
    entry.converted_value_data_type = "audio_path"

    with pytest.raises(ValueError) as excinfo:
        await target._build_chat_messages_for_multi_modal_async([PromptRequestResponse(request_pieces=[entry])])
    assert "Multimodal data type audio_path is not yet supported." in str(excinfo.value)


@pytest.mark.asyncio
async def test_construct_request_body_includes_extra_body_params(
    patch_central_database, dummy_text_request_piece: PromptRequestPiece
):
    target = OpenAIChatTarget(
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
    is_json, target: OpenAIChatTarget, dummy_text_request_piece: PromptRequestPiece
):

    request = PromptRequestResponse(request_pieces=[dummy_text_request_piece])

    body = await target._construct_request_body(conversation=[request], is_json_response=is_json)
    if is_json:
        assert body["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in body


@pytest.mark.asyncio
async def test_construct_request_body_removes_empty_values(
    target: OpenAIChatTarget, dummy_text_request_piece: PromptRequestPiece
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
    target: OpenAIChatTarget, dummy_text_request_piece: PromptRequestPiece
):
    request = PromptRequestResponse(request_pieces=[dummy_text_request_piece])

    body = await target._construct_request_body(conversation=[request], is_json_response=False)
    assert (
        body["messages"][0]["content"] == "dummy text"
    ), "Text messages are serialized in a simple way that's more broadly supported"


@pytest.mark.asyncio
async def test_construct_request_body_serializes_complex_message(
    target: OpenAIChatTarget, dummy_text_request_piece: PromptRequestPiece
):
    request = PromptRequestResponse(request_pieces=[dummy_text_request_piece, get_image_request_piece()])

    body = await target._construct_request_body(conversation=[request], is_json_response=False)
    messages = body["messages"][0]["content"]
    assert len(messages) == 2, "Complex messages are serialized as a list"
    assert messages[0]["type"] == "text", "Text messages are serialized properly when multi-modal"
    assert messages[1]["type"] == "image_url", "Image messages are serialized properly when multi-modal"


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_adds_to_memory(openai_response_json: dict, target: OpenAIChatTarget):
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
    openai_response_json["choices"][0]["message"]["content"] = ""

    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(openai_response_json)

    with patch.object(
        target,
        "_convert_local_image_to_data_url",
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
    target: OpenAIChatTarget,
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
async def test_send_prompt_async_bad_request_error_adds_to_memory(target: OpenAIChatTarget):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()

    target._memory = mock_memory

    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="123", original_value="Hello")]
    )

    response = MagicMock()
    response.status_code = 400

    side_effect = httpx.HTTPStatusError("Bad Request", response=response, request=MagicMock())

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", side_effect=side_effect):
        with pytest.raises(httpx.HTTPStatusError) as bre:
            await target.send_prompt_async(prompt_request=prompt_request)
            target._memory.get_conversation.assert_called_once_with(conversation_id="123")
            target._memory.add_request_response_to_memory.assert_called_once_with(request=prompt_request)

            assert str(bre.value) == "Bad Request"


@pytest.mark.asyncio
async def test_send_prompt_async(openai_response_json: dict, target: OpenAIChatTarget):
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
    with patch.object(
        target,
        "_convert_local_image_to_data_url",
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
async def test_send_prompt_async_empty_response_retries(openai_response_json: dict, target: OpenAIChatTarget):
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
    openai_response_json["choices"][0]["message"]["content"] = ""
    with patch.object(
        target,
        "_convert_local_image_to_data_url",
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
async def test_send_prompt_async_rate_limit_exception_retries(target: OpenAIChatTarget):

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
async def test_send_prompt_async_bad_request_error(target: OpenAIChatTarget):

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
async def test_send_prompt_async_content_filter(target: OpenAIChatTarget):

    response_body = json.dumps(
        {
            "choices": [
                {
                    "content_filter_results": {"violence": {"filtered": True, "severity": "medium"}},
                    "finish_reason": "content_filter",
                    "message": {"content": "Offending content omitted since this is just a test.", "role": "assistant"},
                }
            ],
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
        assert "content_filter_results" in response.get_value()


def test_validate_request_unsupported_data_types(target: OpenAIChatTarget):

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


def test_is_json_response_supported(target: OpenAIChatTarget):
    assert target.is_json_response_supported() is True


def test_is_response_format_json_supported(target: OpenAIChatTarget):

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


def test_is_response_format_json_no_metadata(target: OpenAIChatTarget):
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


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_construct_prompt_response_valid_stop(
    finish_reason: str, target: OpenAIChatTarget, dummy_text_request_piece: PromptRequestPiece
):
    response_dict = {"choices": [{"finish_reason": f"{finish_reason}", "message": {"content": "Hello from stop"}}]}
    response_str = json.dumps(response_dict)

    result = target._construct_prompt_response_from_openai_json(
        open_ai_str_response=response_str, request_piece=dummy_text_request_piece
    )

    assert len(result.request_pieces) == 1
    assert result.get_value() == "Hello from stop"


def test_construct_prompt_response_empty_response(
    target: OpenAIChatTarget, dummy_text_request_piece: PromptRequestPiece
):
    response_dict = {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}
    response_str = json.dumps(response_dict)

    with pytest.raises(EmptyResponseException) as excinfo:
        target._construct_prompt_response_from_openai_json(
            open_ai_str_response=response_str, request_piece=dummy_text_request_piece
        )
    assert "The chat returned an empty response." in str(excinfo.value)


def test_construct_prompt_response_unknown_finish_reason(
    target: OpenAIChatTarget, dummy_text_request_piece: PromptRequestPiece
):
    response_dict = {"choices": [{"finish_reason": "unexpected", "message": {"content": "Some content"}}]}
    response_str = json.dumps(response_dict)

    with pytest.raises(PyritException) as excinfo:
        target._construct_prompt_response_from_openai_json(
            open_ai_str_response=response_str, request_piece=dummy_text_request_piece
        )
    assert "Unknown finish_reason" in str(excinfo.value)


@pytest.mark.asyncio
async def test_openai_chat_target_no_api_version(sample_conversations: MutableSequence[PromptRequestPiece]):
    target = OpenAIChatTarget(
        api_key="test_key", endpoint="https://mock.azure.com", model_name="gpt-35-turbo", api_version=None
    )
    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = MagicMock()
        mock_request.return_value.status_code = 200
        mock_request.return_value.text = '{"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}'

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" not in called_params


@pytest.mark.asyncio
async def test_openai_chat_target_default_api_version(sample_conversations: MutableSequence[PromptRequestPiece]):
    target = OpenAIChatTarget(api_key="test_key", endpoint="https://mock.azure.com", model_name="gpt-35-turbo")
    request_piece = sample_conversations[0]
    request = PromptRequestResponse(request_pieces=[request_piece])

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = MagicMock()
        mock_request.return_value.status_code = 200
        mock_request.return_value.text = '{"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}'

        await target.send_prompt_async(prompt_request=request)

        called_params = mock_request.call_args[1]["params"]
        assert "api-version" in called_params
        assert called_params["api-version"] == "2024-06-01"


@pytest.mark.asyncio
async def test_send_prompt_async_calls_refresh_auth_headers(target: OpenAIChatTarget):
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
            mock_make_request.return_value = MagicMock(
                text='{"choices": [{"finish_reason": "stop", "message": {"content": "test response"}}]}'
            )

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
