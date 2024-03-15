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


# QUESTION: What should this unit test be looking at? It requires ENV Variables to be configured to
# connect to the target.
# Should this be checking that contents are in memory?
def test_send_prompt(azure_blob_storage_target: AzureBlobStorageTarget):
    azure_blob_storage_target.send_prompt(normalized_prompt="data", conversation_id="1", normalizer_id="2")

    # TODO: Check if labels are applied correctly in memory - through manual inspection does not appear to be.


@pytest.mark.asyncio
async def test_send_prompt_async(azure_blob_storage_target: AzureBlobStorageTarget):
    # NOTE: Times out after 6min, with `aiohttp.client_exceptions.ServerTimeoutError: Timeout on
    # reading data from socket`
    await azure_blob_storage_target.send_prompt_async(normalized_prompt="data", conversation_id="1", normalizer_id="2")
