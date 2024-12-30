# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.chat_message_normalizer import ChatMessageNop, GenericSystemSquash
from pyrit.models import ChatMessage


def test_chat_message_nop():
    messages = [
        ChatMessage(role="system", content="System message"),
        ChatMessage(role="user", content="User message 1"),
        ChatMessage(role="assistant", content="Assitant message"),
    ]
    chat_message_nop = ChatMessageNop()
    result = chat_message_nop.normalize(messages)
    assert len(result) == 3
    assert result[0].role == "system"


def test_generic_squash_system_message():
    messages = [
        ChatMessage(role="system", content="System message"),
        ChatMessage(role="user", content="User message 1"),
        ChatMessage(role="assistant", content="Assitant message"),
    ]
    result = GenericSystemSquash().normalize(messages)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].content == "### Instructions ###\n\nSystem message\n\n######\n\nUser message 1"
    assert result[1].role == "assistant"
    assert result[1].content == "Assitant message"


def test_generic_squash_system_message_empty_list():
    with pytest.raises(ValueError):
        GenericSystemSquash().normalize(messages=[])


def test_generic_squash_system_message_single_system_message():
    messages = [ChatMessage(role="system", content="System message")]
    result = GenericSystemSquash().normalize(messages)
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "System message"


def test_generic_squash_system_message_multiple_messages():
    messages = [
        ChatMessage(role="system", content="System message"),
        ChatMessage(role="user", content="User message 1"),
        ChatMessage(role="assistant", content="Assitant message"),
    ]
    result = GenericSystemSquash().normalize(messages)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].content == "### Instructions ###\n\nSystem message\n\n######\n\nUser message 1"
    assert result[1].role == "assistant"
    assert result[1].content == "Assitant message"


def test_generic_squash_system_message_no_system_message():
    messages = [ChatMessage(role="user", content="User message 1"), ChatMessage(role="user", content="User message 2")]
    result = GenericSystemSquash().normalize(messages)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].content == "User message 1"
    assert result[1].role == "user"
    assert result[1].content == "User message 2"
