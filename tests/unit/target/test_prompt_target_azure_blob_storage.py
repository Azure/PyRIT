# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.storage.blob.aio import BlobClient as AsyncBlobClient
from azure.storage.blob.aio import ContainerClient as AsyncContainerClient
from unit.mocks import get_sample_conversations

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import AzureBlobStorageTarget


@pytest.fixture
def sample_entries() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def azure_blob_storage_target(patch_central_database):
    return AzureBlobStorageTarget(
        container_url="https://test.blob.core.windows.net/test",
        sas_token="valid_sas_token",
    )


def test_initialization_with_required_parameters(azure_blob_storage_target: AzureBlobStorageTarget):
    assert azure_blob_storage_target._container_url == "https://test.blob.core.windows.net/test"
    assert azure_blob_storage_target._client_async is None
    assert azure_blob_storage_target._sas_token == "valid_sas_token"


def test_initialization_with_required_parameters_from_env():
    os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE] = (
        "https://test.blob.core.windows.net/test"
    )
    os.environ[AzureBlobStorageTarget.SAS_TOKEN_ENVIRONMENT_VARIABLE] = "valid_sas_token"
    abs_target = AzureBlobStorageTarget()
    assert abs_target._container_url == os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE]
    assert abs_target._sas_token is None


@patch.dict(
    "os.environ",
    {
        AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: "",
    },
)
def test_initialization_with_no_container_url_raises():
    os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE] = ""
    with pytest.raises(ValueError):
        AzureBlobStorageTarget()


@patch("azure.storage.blob.aio.ContainerClient.upload_blob")
@pytest.mark.asyncio
async def test_azure_blob_storage_validate_request_length(
    mock_upload_async,
    azure_blob_storage_target: AzureBlobStorageTarget,
    sample_entries: MutableSequence[PromptRequestPiece],
):
    mock_upload_async.return_value = None
    request = PromptRequestResponse(request_pieces=sample_entries)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await azure_blob_storage_target.send_prompt_async(prompt_request=request)


@patch("azure.storage.blob.aio.ContainerClient.upload_blob")
@pytest.mark.asyncio
async def test_azure_blob_storage_validate_prompt_type(
    mock_upload_async,
    azure_blob_storage_target: AzureBlobStorageTarget,
    sample_entries: MutableSequence[PromptRequestPiece],
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
    mock_upload_async,
    azure_blob_storage_target: AzureBlobStorageTarget,
    sample_entries: MutableSequence[PromptRequestPiece],
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
@patch.object(AzureBlobStorageTarget, "_create_container_client_async", new_callable=AsyncMock)
@patch.object(AsyncBlobClient, "upload_blob", new_callable=AsyncMock)
@patch.object(AsyncContainerClient, "get_blob_client", new_callable=MagicMock)
async def test_send_prompt_async(
    mock_get_blob_client,
    mock_upload_blob,
    mock_create_client,
    azure_blob_storage_target: AzureBlobStorageTarget,
    sample_entries: MutableSequence[PromptRequestPiece],
):
    mock_blob_client = AsyncMock()
    mock_get_blob_client.return_value = mock_blob_client

    mock_blob_client.upload_blob = mock_upload_blob
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
