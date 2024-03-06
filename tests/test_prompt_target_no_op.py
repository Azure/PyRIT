# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib

import pytest

from pyrit.memory import FileMemory
from pyrit.prompt_target import NoOpTarget


@pytest.fixture
def memory(tmp_path: pathlib.Path):
    return FileMemory(filepath=tmp_path / "target_no_op_test.json.memory")


def test_send_prompt_user_no_system(memory: FileMemory):
    no_op = NoOpTarget(memory=memory)

    no_op.send_prompt(
        normalized_prompt="hi, I am a victim chatbot, how can I help?", conversation_id="1", normalizer_id="2"
    )

    chats = no_op.memory.get_memories_with_conversation_id(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "user"
