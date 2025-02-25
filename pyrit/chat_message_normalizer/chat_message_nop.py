# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.chat_message_normalizer import ChatMessageNormalizer
from pyrit.models import ChatMessage


class ChatMessageNop(ChatMessageNormalizer[list[ChatMessage]]):
    def normalize(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """
        returns the same list as was passed in
        """
        return messages
