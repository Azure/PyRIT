# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.message_normalizer import GenericSystemSquashNormalizer
from pyrit.models import Message, MessagePiece


def _make_message(role: str, content: str) -> Message:
    """Helper to create a Message from role and content."""
    return Message(message_pieces=[MessagePiece(role=role, original_value=content)])


@pytest.mark.asyncio
async def test_generic_squash_system_message():
    messages = [
        _make_message("system", "System message"),
        _make_message("user", "User message 1"),
        _make_message("assistant", "Assistant message"),
    ]
    result = await GenericSystemSquashNormalizer().normalize_async(messages)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].get_value() == "### Instructions ###\n\nSystem message\n\n######\n\nUser message 1"
    assert result[1].role == "assistant"
    assert result[1].get_value() == "Assistant message"


@pytest.mark.asyncio
async def test_generic_squash_system_message_empty_list():
    with pytest.raises(ValueError):
        await GenericSystemSquashNormalizer().normalize_async(messages=[])


@pytest.mark.asyncio
async def test_generic_squash_system_message_single_system_message():
    messages = [_make_message("system", "System message")]
    result = await GenericSystemSquashNormalizer().normalize_async(messages)
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].get_value() == "System message"


@pytest.mark.asyncio
async def test_generic_squash_system_message_multiple_messages():
    messages = [
        _make_message("system", "System message"),
        _make_message("user", "User message 1"),
        _make_message("assistant", "Assistant message"),
    ]
    result = await GenericSystemSquashNormalizer().normalize_async(messages)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].get_value() == "### Instructions ###\n\nSystem message\n\n######\n\nUser message 1"
    assert result[1].role == "assistant"
    assert result[1].get_value() == "Assistant message"


@pytest.mark.asyncio
async def test_generic_squash_system_message_no_system_message():
    messages = [_make_message("user", "User message 1"), _make_message("user", "User message 2")]
    result = await GenericSystemSquashNormalizer().normalize_async(messages)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].get_value() == "User message 1"
    assert result[1].role == "user"
    assert result[1].get_value() == "User message 2"
