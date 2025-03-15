# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This module contains the functionality to normalize chat messages into compatible formats for targets.
"""

from pyrit.chat_message_normalizer.chat_message_normalizer import ChatMessageNormalizer
from pyrit.chat_message_normalizer.generic_system_squash import GenericSystemSquash

from pyrit.chat_message_normalizer.chat_message_nop import ChatMessageNop
from pyrit.chat_message_normalizer.chat_message_normalizer_chatml import ChatMessageNormalizerChatML
from pyrit.chat_message_normalizer.chat_message_normalizer_tokenizer import ChatMessageNormalizerTokenizerTemplate

__all__ = [
    "ChatMessageNormalizer",
    "ChatMessageNop",
    "GenericSystemSquash",
    "ChatMessageNormalizerChatML",
    "ChatMessageNormalizerTokenizerTemplate",
]
