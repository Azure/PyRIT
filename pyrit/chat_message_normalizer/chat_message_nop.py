# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import ChatMessage
from pyrit.chat_message_normalizer import ChatMessageNormalizer, NormalizedChatMessage


class ChatMessageNop(ChatMessageNormalizer):
    def normalize(self, messages: list[ChatMessage]) -> NormalizedChatMessage:
        """
        returns the same list as was passed in
        """
        return messages
