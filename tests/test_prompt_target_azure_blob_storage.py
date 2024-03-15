# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import os
import pathlib
import pytest

from pyrit.memory import FileMemory
from pyrit.prompt_target import AzureBlobStorageTarget


@pytest.fixture
def azure_blob_storage_target(tmp_path: pathlib.Path):
    file_memory = FileMemory(filepath=tmp_path / "target_test.json.memory")

    return AzureBlobStorageTarget(
        container_url="https://storage-account-name.blob.core.windows.net/container-name",
        sas_token="valid_sas_token",
        memory=file_memory,
    )


def test_initialization_with_required_parameters(azure_blob_storage_target: AzureBlobStorageTarget):
    assert (
        azure_blob_storage_target._container_url == "https://storage-account-name.blob.core.windows.net/container-name"
    )
    assert azure_blob_storage_target._sas_token == "valid_sas_token"
    assert azure_blob_storage_target._client is not None


@patch.dict(
    "os.environ",
    {
        AzureBlobStorageTarget.SAS_TOKEN_ENVIRONMENT_VARIABLE: "valid_sas_token",
        AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE:
            "https://storage-account-name.blob.core.windows.net/container-name",
    },
)
def test_initialization_with_required_parameters_from_env():
    abs_target = AzureBlobStorageTarget()
    assert abs_target._container_url == os.environ[AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE]
    assert abs_target._sas_token == os.environ[AzureBlobStorageTarget.SAS_TOKEN_ENVIRONMENT_VARIABLE]


@patch.dict(
    "os.environ",
    {
        AzureBlobStorageTarget.SAS_TOKEN_ENVIRONMENT_VARIABLE: "",
        AzureBlobStorageTarget.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE:
            "https://storage-account-name.blob.core.windows.net/container-name",
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
def test_send_prompt(mock_upload, azure_blob_storage_target: AzureBlobStorageTarget):
    mock_upload.return_value = None
    blob_url = azure_blob_storage_target.send_prompt(normalized_prompt=__name__, conversation_id="1", normalizer_id="2")
    assert blob_url.__contains__(azure_blob_storage_target._container_url)

    chats = azure_blob_storage_target._memory.get_memories_with_conversation_id(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "user"
    assert chats[0].content == __name__


@patch("azure.storage.blob.aio.ContainerClient.upload_blob")
@pytest.mark.asyncio
async def test_send_prompt_async(mock_upload_async, azure_blob_storage_target: AzureBlobStorageTarget):
    mock_upload_async.return_value = None
    blob_url = await azure_blob_storage_target.send_prompt_async(
        normalized_prompt=__name__, conversation_id="1", normalizer_id="2"
    )
    assert blob_url.__contains__(azure_blob_storage_target._container_url)

    chats = azure_blob_storage_target._memory.get_memories_with_conversation_id(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "user"
    assert chats[0].content == __name__
