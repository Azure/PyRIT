# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from contextlib import AbstractAsyncContextManager
from unittest.mock import AsyncMock, MagicMock, patch
from tempfile import NamedTemporaryFile
import pytest

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.prompt_target import AzureOpenAIGPTVChatTarget
from pyrit.models.models import ChatMessage


@pytest.fixture
def azure_gptv_chat_engine() -> AzureOpenAIGPTVChatTarget:
    return AzureOpenAIGPTVChatTarget(
        deployment_name="gpt-v",
        endpoint="https://mock.azure.com/",
        api_key="mock-api-key",
        api_version="some_version",
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
        ret = await azure_gptv_chat_engine._complete_chat_async(messages=[ChatMessage(role="user", content=[{"text": "hello"}])])
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
            deployment_name="gpt-4",
            endpoint="",
            api_key="xxxxx",
            api_version="some_version",
            headers=""
        )


def test_convert_image_to_data_url_file_not_found(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    with pytest.raises(FileNotFoundError):
        azure_gptv_chat_engine.convert_local_image_to_data_url("nonexistent.jpg")
        

def test_convert_image_with_unsupported_extension(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    
    with NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name

    assert os.path.exists(tmp_file_name)
    
    with pytest.raises(ValueError) as exc_info:
        azure_gptv_chat_engine.convert_local_image_to_data_url(tmp_file_name)

    assert "Unsupported image format" in str(exc_info.value)

    os.remove(tmp_file_name)
    

@patch('os.path.exists', return_value=True)
@patch('mimetypes.guess_type', return_value=('image/jpg', None))
@patch('pyrit.prompt_normalizer.data_type_serializer.ImagePathDataTypeSerializer')
def test_convert_image_to_data_url_success(mock_serializer_class, mock_guess_type, mock_exists, azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
    mock_serializer_instance = MagicMock()
    mock_serializer_instance.read_data_base64.return_value = 'encoded_base64_string'
    mock_serializer_class.return_value = mock_serializer_instance

    assert os.path.exists(tmp_file_name)
    
    result = azure_gptv_chat_engine.convert_local_image_to_data_url(tmp_file_name)
    assert "data:image/jpeg;base64,encoded_base64_string" in result
    
    # Assertions for the mocks
    mock_serializer_class.assert_called_once_with(prompt_text=tmp_file_name)
    mock_serializer_instance.read_data_base64.assert_called_once()
    
    os.remove(tmp_file_name)


def test_build_chat_messages_with_consistent_roles(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    
    entries = [
        PromptRequestResponse(request_pieces=[
            PromptRequestPiece(role='user', converted_prompt_data_type='text', original_prompt_text="Hello", converted_prompt_text='Hello'),
            PromptRequestPiece(role='user', converted_prompt_data_type='image_path', original_prompt_text='image.jpg', converted_prompt_text='image.jpg')
        ])
    ]
    with patch.object(azure_gptv_chat_engine, 'convert_local_image_to_data_url', return_value='data:image/jpeg;base64,encoded_string'):
        messages = azure_gptv_chat_engine.build_chat_messages(entries)
    
    assert len(messages) == 1
    assert messages[0].role == 'user'
    assert messages[0].content[0]['type'] == 'text'
    assert messages[0].content[1]['type'] == 'image_url'
    

def test_build_chat_messages_with_inconsistent_roles(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    
    entries = [
        PromptRequestResponse(request_pieces=[
            PromptRequestPiece(role='user', converted_prompt_data_type='text', original_prompt_text="Hello", converted_prompt_text='Hello'),
            PromptRequestPiece(role='assistant', converted_prompt_data_type='image_path', original_prompt_text='image.jpg', converted_prompt_text='image.jpg')
        ])
    ]
    with pytest.raises(ValueError) as excinfo:
        azure_gptv_chat_engine.build_chat_messages(entries)
    assert "Inconsistent roles within the same prompt request response entry." in str(excinfo.value)


def test_build_chat_messages_with_unsupported_data_types(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    
    entries = [
        PromptRequestResponse(request_pieces=[
            PromptRequestPiece(role='user', original_prompt_data_type='audio', converted_prompt_data_type='audio', original_prompt_text="audio.mp3", converted_prompt_text='audio.mp3')
        ])
    ]
    with pytest.raises(ValueError) as excinfo:
        azure_gptv_chat_engine.build_chat_messages(entries)
    assert "Multimodal data type audio is not yet supported." in str(excinfo.value)


def test_build_chat_messages_no_roles(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    entries = [
        PromptRequestResponse(request_pieces=[
            PromptRequestPiece(role='', converted_prompt_data_type='text', original_prompt_text='Hello', converted_prompt_text='Hello')
        ])
    ]
    with pytest.raises(ValueError) as excinfo:
        azure_gptv_chat_engine.build_chat_messages(entries)
    assert "No role could be determined from the prompt request pieces." in str(excinfo.value)
    

@pytest.mark.asyncio
async def test_send_prompt_async_successful(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    azure_gptv_chat_engine._memory.get_conversation = MagicMock(return_value=[])
    azure_gptv_chat_engine._memory.add_request_response_to_memory = AsyncMock()
    azure_gptv_chat_engine._memory.add_response_entries_to_memory = AsyncMock()

    azure_gptv_chat_engine._complete_chat_async = AsyncMock(return_value="Mock response text")

    prompt_request = PromptRequestResponse(request_pieces=[PromptRequestPiece(role='user', conversation_id='123', original_prompt_text="Hello")])

    result = await azure_gptv_chat_engine.send_prompt_async(prompt_request=prompt_request)

    azure_gptv_chat_engine._memory.get_conversation.assert_called_once_with(conversation_id='123')
    azure_gptv_chat_engine._memory.add_request_response_to_memory.assert_called_once_with(request=prompt_request)
    azure_gptv_chat_engine._memory.add_response_entries_to_memory.assert_called_once()

    assert result is not None, "Expected a result but got None"
    

@pytest.mark.asyncio
async def test_send_prompt_async_empty_response(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    azure_gptv_chat_engine._memory.get_conversation = MagicMock(return_value=[])
    azure_gptv_chat_engine._memory.add_request_response_to_memory = AsyncMock()
    azure_gptv_chat_engine._memory.add_response_entries_to_memory = AsyncMock()

    azure_gptv_chat_engine._complete_chat_async = AsyncMock(return_value="")

    prompt_request = PromptRequestResponse(request_pieces=[PromptRequestPiece(role='user', original_prompt_text='Hello', conversation_id='123')])

    with pytest.raises(ValueError) as excinfo:
        await azure_gptv_chat_engine.send_prompt_async(prompt_request=prompt_request)

    assert "The chat returned an empty response." in str(excinfo.value), "Expected ValueError for empty response not raised"
    azure_gptv_chat_engine._memory.get_conversation.assert_called_once_with(conversation_id='123')
    

def test_parse_chat_completion_successful(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Test response message"
    result = azure_gptv_chat_engine.parse_chat_completion(mock_response)
    assert result == "Test response message", "The response message was not parsed correctly"


def test_validate_request_too_many_request_pieces(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    
    prompt_request = PromptRequestResponse(request_pieces=[
        PromptRequestPiece(role='user', original_prompt_text='Hello', converted_prompt_data_type='text'),
        PromptRequestPiece(role='user', original_prompt_text='Hello', converted_prompt_data_type='text'),
        PromptRequestPiece(role='user', original_prompt_text='Hello', converted_prompt_data_type='text')
    ])
    with pytest.raises(ValueError) as excinfo:
        azure_gptv_chat_engine.validate_request(prompt_request=prompt_request)

    assert "two prompt request pieces" in str(excinfo.value), "Error not raised for too many request pieces"


def test_validate_request_unsupported_data_types(azure_gptv_chat_engine: AzureOpenAIGPTVChatTarget):
    prompt_request = PromptRequestResponse(request_pieces=[
        PromptRequestPiece(role='user', original_prompt_text='Hello', converted_prompt_data_type='text'),
        PromptRequestPiece(role='user', original_prompt_text='Hello', converted_prompt_data_type='image_path'),
        PromptRequestPiece(role='user', original_prompt_text='Hello', converted_prompt_data_type='video_path')
    ])

    with pytest.raises(ValueError) as excinfo:
        azure_gptv_chat_engine.validate_request(prompt_request=prompt_request)

    assert "two prompt request pieces text and image_path." in str(excinfo.value), "Error not raised for unsupported data types"
