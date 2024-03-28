# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from tempfile import NamedTemporaryFile
from typing import Generator
import pytest

from pyrit.memory import MemoryInterface
from pyrit.prompt_target import TextTarget

from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


def test_send_prompt_user_no_system(memory_interface: MemoryInterface):
    no_op = TextTarget(memory=memory_interface)

    no_op.send_prompt(
        normalized_prompt="hi, I am a victim chatbot, how can I help?", conversation_id="1", normalizer_id="2"
    )

    chats = no_op._memory.get_prompt_entries_with_conversation_id(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "user"


def test_send_prompt_stream(memory_interface: MemoryInterface):
    with NamedTemporaryFile(mode="w+", delete=False) as tmp_file:
        prompt = "hi, I am a victim chatbot, how can I help?"

        no_op = TextTarget(memory=memory_interface, text_stream=tmp_file)
        no_op.send_prompt(normalized_prompt=prompt, conversation_id="1", normalizer_id="2")

        tmp_file.seek(0)
        content = tmp_file.read()

    os.remove(tmp_file.name)

    assert prompt in content, "The prompt was not found in the temporary file content."
