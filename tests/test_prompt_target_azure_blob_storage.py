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
    # TODO: Change content to make some assertions,
    # Maybe also check memory here
    azure_blob_storage_target.send_prompt(normalized_prompt="test", conversation_id="1", normalizer_id="2")
    print(azure_blob_storage_target.list_created_blob_urls()
    print(azure_blob_storage_target.list_all_blob_urls())
