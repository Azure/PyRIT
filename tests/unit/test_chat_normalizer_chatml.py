# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import textwrap

import pytest

from pyrit.chat_message_normalizer import ChatMessageNormalizerChatML
from pyrit.models import ChatMessage


@pytest.fixture
def normalizer():
    return ChatMessageNormalizerChatML()


def test_normalize(normalizer: ChatMessageNormalizerChatML):
    messages = [ChatMessage(role="user", content="Hello"), ChatMessage(role="assistant", content="Hi there!")]
    expected = textwrap.dedent(
        """\
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
        """
    )

    assert normalizer.normalize(messages) == expected


def test_normalize_with_name(normalizer: ChatMessageNormalizerChatML):
    messages = [
        ChatMessage(role="user", content="Hello", name="user001"),
        ChatMessage(role="assistant", content="Hi there!"),
    ]
    expected = textwrap.dedent(
        """\
        <|im_start|>user name=user001
        Hello<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
        """
    )

    assert normalizer.normalize(messages) == expected


def test_from_chatml_raises_error_when_invalid(normalizer: ChatMessageNormalizerChatML):
    with pytest.raises(ValueError):
        normalizer.from_chatml("asdf")


def test_from_chatml(normalizer: ChatMessageNormalizerChatML):
    chatml = textwrap.dedent(
        """\
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
        """
    )

    expected = [ChatMessage(role="user", content="Hello"), ChatMessage(role="assistant", content="Hi there!")]

    assert normalizer.from_chatml(chatml) == expected


def test_from_chatml_with_name(normalizer: ChatMessageNormalizerChatML):
    chatml = textwrap.dedent(
        """\
        <|im_start|>user name=user001
        Hello<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
        """
    )
    expected = [
        ChatMessage(role="user", content="Hello", name="user001"),
        ChatMessage(role="assistant", content="Hi there!"),
    ]

    assert normalizer.from_chatml(chatml) == expected


def test_from_chatml_raises_error_when_empty(normalizer: ChatMessageNormalizerChatML):
    with pytest.raises(ValueError):
        normalizer.from_chatml("n/a")
