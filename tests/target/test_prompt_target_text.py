# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import os
from tempfile import NamedTemporaryFile
from typing import Generator
from unittest.mock import patch
import pytest

from pyrit.memory import MemoryInterface
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse
from pyrit.prompt_target import TextTarget

from tests.mocks import get_memory_interface
from tests.mocks import get_sample_conversations


@pytest.fixture
def sample_entries() -> list[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.asyncio
async def test_send_prompt_user_no_system(memory_interface: MemoryInterface, sample_entries: list[PromptRequestPiece]):
    output_stream = io.StringIO()
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        no_op = TextTarget(text_stream=output_stream)

        request = sample_entries[0]
        request.converted_value = "hi, I am a victim chatbot, how can I help?"

        await no_op.send_prompt_async(prompt_request=PromptRequestResponse(request_pieces=[request]))

        output_stream.seek(0)
        captured_output = output_stream.read()

        assert captured_output
        assert request.converted_value in captured_output


@pytest.mark.asyncio
async def test_send_prompt_stream(memory_interface: MemoryInterface, sample_entries: list[PromptRequestPiece]):
    with NamedTemporaryFile(mode="w+", delete=False) as tmp_file:
        prompt = "hi, I am a victim chatbot, how can I help?"
        with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
            no_op = TextTarget(text_stream=tmp_file)
            request = sample_entries[0]
            request.converted_value = prompt

            await no_op.send_prompt_async(prompt_request=PromptRequestResponse(request_pieces=[request]))

            tmp_file.seek(0)
            content = tmp_file.read()

    os.remove(tmp_file.name)

    assert prompt in content, "The prompt was not found in the temporary file content."
