# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pathlib
import pytest
from tempfile import NamedTemporaryFile

from pyrit.memory import FileMemory
from pyrit.prompt_target import TextTarget


@pytest.fixture
def memory(tmp_path: pathlib.Path):
    return FileMemory(filepath=tmp_path / "target_no_op_test.json.memory")


def test_send_prompt_user_no_system(memory: FileMemory):
    no_op = TextTarget(memory=memory)

    no_op.send_prompt(
        normalized_prompt="hi, I am a victim chatbot, how can I help?", conversation_id="1", normalizer_id="2"
    )

    chats = no_op.memory.get_memories_with_conversation_id(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "user"


def test_send_prompt_stream(memory: FileMemory):
    with NamedTemporaryFile(mode="w+", delete=False) as tmp_file:
        prompt = "hi, I am a victim chatbot, how can I help?"

        no_op = TextTarget(memory=memory, text_stream=tmp_file)
        no_op.send_prompt(normalized_prompt=prompt, conversation_id="1", normalizer_id="2")

        tmp_file.seek(0)
        content = tmp_file.read()

    os.remove(tmp_file.name)

    assert prompt in content, "The prompt was not found in the temporary file content."
