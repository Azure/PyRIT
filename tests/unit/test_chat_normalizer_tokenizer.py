# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import textwrap

import pytest
from transformers import AutoTokenizer

from pyrit.chat_message_normalizer import ChatMessageNormalizerTokenizerTemplate
from pyrit.models import ChatMessage


@pytest.fixture
def blenderbot_tokenizer_normalizer():
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    return ChatMessageNormalizerTokenizerTemplate(tokenizer)


@pytest.fixture
def chatml_tokenizer_normalizer():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    return ChatMessageNormalizerTokenizerTemplate(tokenizer)


@pytest.mark.skip(reason="This is broken running in github. Internal work-item 2587")
def test_normalize_blenderbot(blenderbot_tokenizer_normalizer: ChatMessageNormalizerTokenizerTemplate):
    messages = [
        ChatMessage(role="user", content="Hello, how are you?"),
        ChatMessage(role="assistant", content="I'm doing great. How can I help you today?"),
        ChatMessage(role="user", content="I'd like to show off how chat templating works!"),
    ]
    expected = (
        " Hello, how are you?  I'm doing great. "
        "How can I help you today?   "
        "I'd like to show off how chat templating works!</s>"
    )

    assert blenderbot_tokenizer_normalizer.normalize(messages) == expected


@pytest.mark.skip(reason="This is broken running in github. Internal work item 2587")
def test_normalize_chatml(chatml_tokenizer_normalizer: ChatMessageNormalizerTokenizerTemplate):
    messages = [
        ChatMessage(role="system", content="You are a friendly chatbot who always responds in the style of a pirate"),
        ChatMessage(role="user", content="How many helicopters can a human eat in one sitting?"),
    ]
    expected = textwrap.dedent(
        """\
        <|system|>
        You are a friendly chatbot who always responds in the style of a pirate</s>
        <|user|>
        How many helicopters can a human eat in one sitting?</s>
        <|assistant|>
        """
    )

    assert chatml_tokenizer_normalizer.normalize(messages) == expected
