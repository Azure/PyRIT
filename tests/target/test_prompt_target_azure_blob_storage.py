# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Generator
import pytest

from unittest.mock import patch

from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.prompt_target import AzureBlobStorageTarget

from tests.mocks import get_memory_interface
from tests.mocks import get_sample_conversations


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def sample_entries() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def azure_blob_storage_target(memory_interface: MemoryInterface):
    return AzureBlobStorageTarget(
        container_url="https://test.blob.core.windows.net/test",
        sas_token="valid_sas_token",
        memory=memory_interface,
    )


def test_initialization_with_required_parameters(azure_blob_storage_target: AzureBlobStorageTarget):
    assert azure_blob_storage_target._container_url == "https://test.blob.core.windows.net/test"
    assert azure_blob_storage_target._sas_token == "valid_sas_token"
    assert azure_blob_storage_target._client is not None


def test_initialization_with_required_parameters_from_env():
    os.environ[AzureBlobStorageTarget.SAS_TOKEN_ENVIRONMENT_VARIABLE] = "valid_sas_token"
    os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE] = (
        "https://test.blob.core.windows.net/test"
    )
    abs_target = AzureBlobStorageTarget()
    assert abs_target._container_url == os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE]
    assert abs_target._sas_token == os.environ[AzureBlobStorageTarget.SAS_TOKEN_ENVIRONMENT_VARIABLE]


@patch.dict(
    "os.environ",
    {
        AzureBlobStorageTarget.SAS_TOKEN_ENVIRONMENT_VARIABLE: "",
        AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: "https://test.blob.core.windows.net/test",
    },
)
def test_initialization_with_no_sas_token_raises():
    with pytest.raises(ValueError):
        AzureBlobStorageTarget(
            container_url=os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE]
        )


@patch.dict(
    "os.environ",
    {
        AzureBlobStorageTarget.SAS_TOKEN_ENVIRONMENT_VARIABLE: "valid_sas_token",
        AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: "",
    },
)
def test_initialization_with_no_container_url_raises():
    os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureBlobStorageTarget(sas_token=os.environ[AzureBlobStorageTarget.SAS_TOKEN_ENVIRONMENT_VARIABLE])


@patch("azure.storage.blob.ContainerClient.upload_blob")
def test_send_prompt(
    mock_upload, azure_blob_storage_target: AzureBlobStorageTarget, sample_entries: list[PromptRequestPiece]
):
    mock_upload.return_value = None
    request_piece = sample_entries[0]
    conversation_id = request_piece.conversation_id
    request_piece.converted_prompt_text = __name__
    request = PromptRequestResponse([request_piece])

    response = azure_blob_storage_target.send_prompt(prompt_request=request)

    assert response

    blob_url = response.request_pieces[0].converted_prompt_text

    assert blob_url.__contains__(azure_blob_storage_target._container_url)
    assert blob_url.__contains__(".txt")

    chats = azure_blob_storage_target._memory.get_prompt_entries_with_conversation_id(conversation_id=conversation_id)
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "user"
    assert azure_blob_storage_target._container_url in chats[0].converted_prompt_text


@patch("azure.storage.blob.aio.ContainerClient.upload_blob")
@pytest.mark.asyncio
async def test_send_prompt_async(
    mock_upload_async, azure_blob_storage_target: AzureBlobStorageTarget, sample_entries: list[PromptRequestPiece]
):
    mock_upload_async.return_value = None
    request_piece = sample_entries[0]
    conversation_id = request_piece.conversation_id
    request_piece.converted_prompt_text = __name__
    request = PromptRequestResponse([request_piece])

    response = await azure_blob_storage_target.send_prompt_async(prompt_request=request)

    assert response

    blob_url = response.request_pieces[0].converted_prompt_text

    assert blob_url.__contains__(azure_blob_storage_target._container_url)
    assert blob_url.__contains__(".txt")

    chats = azure_blob_storage_target._memory.get_prompt_entries_with_conversation_id(conversation_id=conversation_id)
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "user"
    assert azure_blob_storage_target._container_url in chats[0].converted_prompt_text
