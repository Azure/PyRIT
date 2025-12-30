# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Functionality to normalize messages into compatible formats for targets.
"""

from pyrit.message_normalizer.message_normalizer import MessageListNormalizer, MessageStringNormalizer
from pyrit.message_normalizer.generic_system_squash import GenericSystemSquashNormalizer
from pyrit.message_normalizer.message_nop import MessageNop
from pyrit.message_normalizer.chatml_normalizer import ChatMLNormalizer
from pyrit.message_normalizer.tokenizer_template_normalizer import TokenizerTemplateNormalizer
from pyrit.message_normalizer.conversation_context_normalizer import ConversationContextNormalizer
from pyrit.message_normalizer.chat_message_normalizer import ChatMessageNormalizer

__all__ = [
    "MessageListNormalizer",
    "MessageStringNormalizer",
    "MessageNop",
    "GenericSystemSquashNormalizer",
    "ChatMLNormalizer",
    "TokenizerTemplateNormalizer",
    "ConversationContextNormalizer",
    "ChatMessageNormalizer",
]
