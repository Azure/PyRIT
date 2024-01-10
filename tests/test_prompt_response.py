# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
from pathlib import Path

import pytest

from pyrit.models import PromptResponse


@pytest.fixture
def prompt_response_1() -> PromptResponse:
    return PromptResponse(
        completion="This is a test",
        prompt="This is a test",
        id="1234",
        completion_tokens=1,
        prompt_tokens=1,
        total_tokens=1,
        model="test",
        object="test",
        created_at=1,
        logprobs=True,
        index=1,
        finish_reason="test",
        api_request_time_to_complete_ns=1,
    )


def test_saving_of_prompt_response(prompt_response_1: PromptResponse) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        new_file = prompt_response_1.save_to_file(directory_path=Path(tmp_dir))
        assert new_file


def test_save_and_load_of_prompt_response(prompt_response_1: PromptResponse) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save file
        new_file = prompt_response_1.save_to_file(directory_path=Path(tmp_dir))

        # Load file
        loaded_prompt_response = PromptResponse.load_from_file(file_path=Path(new_file))
        assert loaded_prompt_response == prompt_response_1
