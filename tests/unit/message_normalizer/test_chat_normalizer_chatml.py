# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import textwrap

import pytest

from pyrit.message_normalizer import ChatMLNormalizer
from pyrit.models import Message, MessagePiece


def _make_message(role: str, content: str) -> Message:
    """Helper to create a Message from role and content."""
    return Message(message_pieces=[MessagePiece(role=role, original_value=content)])


@pytest.fixture
def normalizer():
    return ChatMLNormalizer()


@pytest.mark.asyncio
async def test_normalize(normalizer: ChatMLNormalizer):
    messages = [_make_message("user", "Hello"), _make_message("assistant", "Hi there!")]
    expected = textwrap.dedent(
        """\
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
        """
    )

    assert await normalizer.normalize_string_async(messages) == expected


@pytest.mark.asyncio
async def test_normalize_with_system(normalizer: ChatMLNormalizer):
    messages = [
        _make_message("system", "You are a helpful assistant."),
        _make_message("user", "Hello"),
        _make_message("assistant", "Hi there!"),
    ]
    expected = textwrap.dedent(
        """\
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
        """
    )

    assert await normalizer.normalize_string_async(messages) == expected


def test_from_chatml_raises_error_when_invalid(normalizer: ChatMLNormalizer):
    with pytest.raises(ValueError):
        normalizer.from_chatml("asdf")


def test_from_chatml(normalizer: ChatMLNormalizer):
    chatml = textwrap.dedent(
        """\
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
        """
    )

    result = normalizer.from_chatml(chatml)
    assert len(result) == 2
    assert result[0].role == "user"
    assert result[0].get_value() == "Hello"
    assert result[1].role == "assistant"
    assert result[1].get_value() == "Hi there!"


def test_from_chatml_raises_error_when_empty(normalizer: ChatMLNormalizer):
    with pytest.raises(ValueError):
        normalizer.from_chatml("n/a")
