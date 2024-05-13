# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from contextlib import AbstractAsyncContextManager
from unittest.mock import AsyncMock, MagicMock, patch
from tempfile import NamedTemporaryFile

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.prompt_target import AzureOpenAIGPTVChatTarget
from pyrit.models import ChatMessageListContent

from tests.mocks import get_image_request_piece


@pytest.fixture
def azure_gptv_chat_engine() -> AzureOpenAIGPTVChatTarget:
    return AzureOpenAIGPTVChatTarget(
        deployment_name="gpt-v",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        api_version="some_version",
    )


@pytest.fixture
def azure_openai_mock_return() -> ChatCompletion:
    return ChatCompletion(
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
        model="gpt-4-v",
    )


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


@patch(
    "openai.resources.chat.AsyncCompletions.create",
    new_callable=lambda: MockChatCompletionsAsync(),
)
@pytest.mark.asyncio
async def test_complete_chat_async_return(
    openai_mock_return: ChatCompletion, azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget
):
    with patch("openai.resources.chat.Completions.create") as mock_create:
        mock_create.return_value = openai_mock_return
        ret = await azure_gptv_chat_engine._complete_chat_async(
            messages=[ChatMessageListContent(role="user", content=[{"text": "hello"}])]
        )
        assert ret == "hi"


def test_init_with_no_api_key_var_raises():
    os.environ[AzureOpenAIGPTVChatTarget.API_KEY_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAIGPTVChatTarget(
            deployment_name="gpt-4",
            endpoint="https://mock.azure.com/",
            api_key="",
            api_version="some_version",
        )


def test_init_with_no_deployment_var_raises():
    os.environ[AzureOpenAIGPTVChatTarget.DEPLOYMENT_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAIGPTVChatTarget()


def test_init_with_no_endpoint_uri_var_raises():
    os.environ[AzureOpenAIGPTVChatTarget.ENDPOINT_URI_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureOpenAIGPTVChatTarget(
            deployment_name="gpt-4",
            endpoint="",
            api_key="xxxxx",
            api_version="some_version",
        )


def test_init_with_no_additional_request_headers_var_raises():
    os.environ[AzureOpenAIGPTVChatTarget.ADDITIONAL_REQUEST_HEADERS] = ""
    with pytest.raises(ValueError):
        AzureOpenAIGPTVChatTarget(
            deployment_name="gpt-4", endpoint="", api_key="xxxxx", api_version="some_version", headers=""
        )


def test_convert_image_to_data_url_file_not_found(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    with pytest.raises(FileNotFoundError):
        azure_gptv_chat_engine._convert_local_image_to_data_url("nonexistent.jpg")


def test_convert_image_with_unsupported_extension(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):

    with NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name

    assert os.path.exists(tmp_file_name)

    with pytest.raises(ValueError) as exc_info:
        azure_gptv_chat_engine._convert_local_image_to_data_url(tmp_file_name)

    assert "Unsupported image format" in str(exc_info.value)

    os.remove(tmp_file_name)


@patch("os.path.exists", return_value=True)
@patch("mimetypes.guess_type", return_value=("image/jpg", None))
@patch("pyrit.models.data_type_serializer.ImagePathDataTypeSerializer")
def test_convert_image_to_data_url_success(
    mock_serializer_class, mock_guess_type, mock_exists, azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget
):
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
    mock_serializer_instance = MagicMock()
    mock_serializer_instance.read_data_base64.return_value = "encoded_base64_string"
    mock_serializer_class.return_value = mock_serializer_instance

    assert os.path.exists(tmp_file_name)

    result = azure_gptv_chat_engine._convert_local_image_to_data_url(tmp_file_name)
    assert "data:image/jpeg;base64,encoded_base64_string" in result

    # Assertions for the mocks
    mock_serializer_class.assert_called_once_with(prompt_text=tmp_file_name)
    mock_serializer_instance.read_data_base64.assert_called_once()

    os.remove(tmp_file_name)


def test_build_chat_messages_with_consistent_roles(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):

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
        azure_gptv_chat_engine, "_convert_local_image_to_data_url", return_value="data:image/jpeg;base64,encoded_string"
    ):
        messages = azure_gptv_chat_engine._build_chat_messages(entries)

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content[0]["type"] == "text"  # type: ignore
    assert messages[0].content[1]["type"] == "image_url"  # type: ignore

    os.remove(image_request.original_value)


def test_build_chat_messages_with_unsupported_data_types(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    # Like an image_path, the audio_path requires a file, but doesn't validate any contents
    entry = get_image_request_piece()
    entry.converted_value_data_type = "audio_path"

    with pytest.raises(ValueError) as excinfo:
        azure_gptv_chat_engine._build_chat_messages([PromptRequestResponse(request_pieces=[entry])])
    assert "Multimodal data type audio_path is not yet supported." in str(excinfo.value)


def test_build_chat_messages_no_roles(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    entries = [
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="",  # type: ignore
                    converted_value_data_type="text",
                    original_value="Hello",
                    converted_value="Hello",
                )
            ]
        )
    ]
    with pytest.raises(ValueError) as excinfo:
        azure_gptv_chat_engine._build_chat_messages(entries)
    assert "No role could be determined from the prompt request pieces." in str(excinfo.value)


@pytest.mark.asyncio
async def test_send_prompt_async_adds_to_memory(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()
    mock_memory.add_response_entries_to_memory = AsyncMock()

    azure_gptv_chat_engine._memory = mock_memory

    mock_complete_chat_async = AsyncMock(return_value="Mock response text")

    mock_complete_chat_async = AsyncMock(return_value="Mock response text")
    setattr(azure_gptv_chat_engine, "_complete_chat_async", mock_complete_chat_async)

    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", conversation_id="123", original_value="Hello")]
    )

    result = await azure_gptv_chat_engine.send_prompt_async(prompt_request=prompt_request)

    azure_gptv_chat_engine._memory.get_conversation.assert_called_once_with(conversation_id="123")
    azure_gptv_chat_engine._memory.add_request_response_to_memory.assert_called_once_with(request=prompt_request)
    azure_gptv_chat_engine._memory.add_response_entries_to_memory.assert_called_once()

    assert result is not None, "Expected a result but got None"


@pytest.mark.asyncio
async def test_send_prompt_async(
    azure_openai_mock_return: ChatCompletion, azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget
):
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
        azure_gptv_chat_engine, "_convert_local_image_to_data_url", return_value="data:image/jpeg;base64,encoded_string"
    ):
        with patch("openai.resources.chat.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = azure_openai_mock_return
            response: PromptRequestResponse = await azure_gptv_chat_engine.send_prompt_async(
                prompt_request=prompt_req_resp
            )
            assert len(response.request_pieces) == 1
            assert response.request_pieces[0].converted_value == "hi"
    os.remove(tmp_file_name)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    mock_memory = MagicMock()
    mock_memory.get_conversation.return_value = []
    mock_memory.add_request_response_to_memory = AsyncMock()
    mock_memory.add_response_entries_to_memory = AsyncMock()
    azure_gptv_chat_engine._memory = mock_memory

    mock_complete_chat_async = AsyncMock(return_value="")
    setattr(azure_gptv_chat_engine, "_complete_chat_async", mock_complete_chat_async)

    prompt_request = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", original_value="Hello", conversation_id="123")]
    )

    with pytest.raises(ValueError) as excinfo:
        await azure_gptv_chat_engine.send_prompt_async(prompt_request=prompt_request)

    assert "The chat returned an empty response." in str(
        excinfo.value
    ), "Expected ValueError for empty response not raised"
    azure_gptv_chat_engine._memory.get_conversation.assert_called_once_with(conversation_id="123")


def test_parse_chat_completion_successful(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Test response message"
    result = azure_gptv_chat_engine._parse_chat_completion(mock_response)
    assert result == "Test response message", "The response message was not parsed correctly"


def test_validate_request_too_many_request_pieces(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):

    prompt_request = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="user", original_value="Hello", converted_value_data_type="text"),
            PromptRequestPiece(role="user", original_value="Hello", converted_value_data_type="text"),
            PromptRequestPiece(role="user", original_value="Hello", converted_value_data_type="text"),
        ]
    )
    with pytest.raises(ValueError) as excinfo:
        azure_gptv_chat_engine._validate_request(prompt_request=prompt_request)

    assert "two prompt request pieces" in str(excinfo.value), "Error not raised for too many request pieces"


def test_validate_request_unsupported_data_types(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):

    image_piece = get_image_request_piece()
    image_piece.converted_value_data_type = "new_unknown_type"  # type: ignore
    prompt_request = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="user", original_value="Hello", converted_value_data_type="text"),
            image_piece,
        ]
    )

    with pytest.raises(ValueError) as excinfo:
        azure_gptv_chat_engine._validate_request(prompt_request=prompt_request)

    assert "This target only supports text and image_path." in str(
        excinfo.value
    ), "Error not raised for unsupported data types"

    os.remove(image_piece.original_value)
