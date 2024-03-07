# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib

import pytest

from pyrit.memory import FileMemory
from pyrit.prompt_target import AzureBlobStorageTarget


@pytest.fixture
def azure_blob_storage_target(tmp_path: pathlib.Path):
    file_memory = FileMemory(filepath=tmp_path / "target_test.json.memory")

    return AzureBlobStorageTarget(
        container_url=None,
        sas_token=None,
        memory=file_memory,
    )

def test_send_prompt(azure_blob_storage_target: AzureBlobStorageTarget):
    azure_blob_storage_target.send_prompt(
        normalized_prompt="test",
        conversation_id="1",
        normalizer_id="2")
    
    # TODO: play around with this iterator, and make sure entries are being added to create blob URL
    # Already validated connecting and uploading file to storage account container
    print(azure_blob_storage_target.list_blob_names())
    
    for entry in azure_blob_storage_target.created_blob_url_list:
        print(entry)