# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from tempfile import NamedTemporaryFile
from typing import Generator
import pytest

from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_piece import PromptRequestPiece
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


def test_send_prompt_user_no_system(memory_interface: MemoryInterface, sample_entries: list[PromptRequestPiece]):
    no_op = TextTarget(memory=memory_interface)

    request = sample_entries[0]
    request.converted_value = "hi, I am a victim chatbot, how can I help?"

    no_op.send_prompt(prompt_request=PromptRequestResponse(request_pieces=[request]))

    chats = no_op._memory._get_prompt_pieces_with_conversation_id(conversation_id=request.conversation_id)
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "user"


def test_send_prompt_stream(memory_interface: MemoryInterface, sample_entries: list[PromptRequestPiece]):
    with NamedTemporaryFile(mode="w+", delete=False) as tmp_file:
        prompt = "hi, I am a victim chatbot, how can I help?"

        no_op = TextTarget(memory=memory_interface, text_stream=tmp_file)
        request = sample_entries[0]
        request.converted_value = prompt

        no_op.send_prompt(prompt_request=PromptRequestResponse(request_pieces=[request]))

        tmp_file.seek(0)
        content = tmp_file.read()

    os.remove(tmp_file.name)

    assert prompt in content, "The prompt was not found in the temporary file content."
