# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.common.chat_message_normalizer import squash_system_message, ChatMessage
import pytest

def test_squash_system_message_empty_list():
    with pytest.raises(ValueError):
        squash_system_message([])

def test_squash_system_message_single_system_message():
    messages = [ChatMessage(role="system", content="System message")]
    result = squash_system_message(messages)
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "System message"

def test_squash_system_message_multiple_messages():
    messages = [
        ChatMessage(role="system", content="System message"),
        ChatMessage(role="user", content="User message 1"),
        ChatMessage(role="assistant", content="Assitant message")
    ]
    result = squash_system_message(messages)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].content == '### Instructions ###\n\nSystem message\n\n######\n\nUser message 1'
    assert result[1].role == "assistant"
    assert result[1].content == "Assitant message"

def test_squash_system_message_no_system_message():
    messages = [
        ChatMessage(role="user", content="User message 1"),
        ChatMessage(role="user", content="User message 2")
    ]
    result = squash_system_message(messages)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].content == "User message 1"
    assert result[1].role == "user"
    assert result[1].content == "User message 2"