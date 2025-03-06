# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import os
from tempfile import NamedTemporaryFile
from typing import MutableSequence

import pytest
from unit.mocks import get_sample_conversations

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import TextTarget


@pytest.fixture
def sample_entries() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.mark.asyncio
async def test_send_prompt_user_no_system(sample_entries: MutableSequence[PromptRequestPiece]):
    output_stream = io.StringIO()
    no_op = TextTarget(text_stream=output_stream)

    request = sample_entries[0]
    request.converted_value = "hi, I am a victim chatbot, how can I help?"

    await no_op.send_prompt_async(prompt_request=PromptRequestResponse(request_pieces=[request]))

    output_stream.seek(0)
    captured_output = output_stream.read()

    assert captured_output
    assert request.converted_value in captured_output


@pytest.mark.asyncio
async def test_send_prompt_stream(sample_entries: MutableSequence[PromptRequestPiece]):
    with NamedTemporaryFile(mode="w+", delete=False) as tmp_file:
        prompt = "hi, I am a victim chatbot, how can I help?"
        no_op = TextTarget(text_stream=tmp_file)
        request = sample_entries[0]
        request.converted_value = prompt

        await no_op.send_prompt_async(prompt_request=PromptRequestResponse(request_pieces=[request]))

        tmp_file.seek(0)
        content = tmp_file.read()

    os.remove(tmp_file.name)

    assert prompt in content, "The prompt was not found in the temporary file content."
