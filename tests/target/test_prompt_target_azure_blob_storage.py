# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Generator
import pytest

from unittest.mock import AsyncMock, patch
from azure.storage.blob.aio import ContainerClient as AsyncContainerClient

from pyrit.memory import MemoryInterface
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse
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
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        return AzureBlobStorageTarget(
            container_url="https://test.blob.core.windows.net/test",
            sas_token="valid_sas_token",
        )


def test_initialization_with_required_parameters(azure_blob_storage_target: AzureBlobStorageTarget):
    assert azure_blob_storage_target._container_url == "https://test.blob.core.windows.net/test"
    assert azure_blob_storage_target._client_async is None
    assert azure_blob_storage_target._sas_token == "valid_sas_token"


def test_initialization_with_required_parameters_from_env(memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE] = (
            "https://test.blob.core.windows.net/test"
        )
        os.environ[AzureBlobStorageTarget.SAS_TOKEN_ENVIRONMENT_VARIABLE] = "valid_sas_token"
        abs_target = AzureBlobStorageTarget()
        assert (
            abs_target._container_url == os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE]
        )
        assert abs_target._sas_token is None


@patch.dict(
    "os.environ",
    {
        AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: "",
    },
)
def test_initialization_with_no_container_url_raises(memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE] = ""
        with pytest.raises(ValueError):
            AzureBlobStorageTarget()


@patch("azure.storage.blob.aio.ContainerClient.upload_blob")
@pytest.mark.asyncio
async def test_azure_blob_storage_validate_request_length(
    mock_upload_async, azure_blob_storage_target: AzureBlobStorageTarget, sample_entries: list[PromptRequestPiece]
):
    mock_upload_async.return_value = None
    request = PromptRequestResponse(request_pieces=sample_entries)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await azure_blob_storage_target.send_prompt_async(prompt_request=request)


@patch("azure.storage.blob.aio.ContainerClient.upload_blob")
@pytest.mark.asyncio
async def test_azure_blob_storage_validate_prompt_type(
    mock_upload_async, azure_blob_storage_target: AzureBlobStorageTarget, sample_entries: list[PromptRequestPiece]
):
    mock_upload_async.return_value = None
    request_piece = sample_entries[0]
    request_piece.converted_value_data_type = "image_path"
    request = PromptRequestResponse(request_pieces=[request_piece])
    with pytest.raises(ValueError, match="This target only supports text and url prompt input."):
        await azure_blob_storage_target.send_prompt_async(prompt_request=request)


@patch("azure.storage.blob.aio.ContainerClient.upload_blob")
@pytest.mark.asyncio
async def test_azure_blob_storage_validate_prev_convs(
    mock_upload_async, azure_blob_storage_target: AzureBlobStorageTarget, sample_entries: list[PromptRequestPiece]
):
    mock_upload_async.return_value = None
    request_piece = sample_entries[0]
    azure_blob_storage_target._memory.add_request_response_to_memory(
        request=PromptRequestResponse(request_pieces=[request_piece])
    )
    request = PromptRequestResponse(request_pieces=[request_piece])

    with pytest.raises(ValueError, match="This target only supports a single turn conversation."):
        await azure_blob_storage_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
@patch.object(AsyncContainerClient, "upload_blob", new_callable=AsyncMock)
@patch.object(AzureBlobStorageTarget, "_create_container_client_async", new_callable=AsyncMock)
async def test_send_prompt_async(
    mock_create_client,
    mock_upload_blob,
    azure_blob_storage_target: AzureBlobStorageTarget,
    sample_entries: list[PromptRequestPiece],
):
    mock_upload_blob.return_value = None
    azure_blob_storage_target._client_async = AsyncContainerClient.from_container_url(
        container_url=azure_blob_storage_target._container_url, credential="mocked_sas_token"
    )

    request_piece = sample_entries[0]
    request_piece.converted_value = "Test content"
    request = PromptRequestResponse([request_piece])

    response = await azure_blob_storage_target.send_prompt_async(prompt_request=request)
    assert response
    blob_url = response.request_pieces[0].converted_value
    assert azure_blob_storage_target._container_url in blob_url
    assert blob_url.endswith(".txt")
    mock_upload_blob.assert_awaited_once()
