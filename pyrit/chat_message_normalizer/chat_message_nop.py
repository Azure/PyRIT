# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.chat_message_normalizer import ChatMessageNormalizer
from pyrit.models import ChatMessage


class ChatMessageNop(ChatMessageNormalizer[list[ChatMessage]]):
    """
    A no-op chat message normalizer that does not modify the input messages.
    """

    def normalize(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """
        Returns the same list as was passed in.

        Args:
            messages (list[ChatMessage]): The list of messages to normalize.

        Returns:
            list[ChatMessage]: The normalized list of messages.
        """
        return messages
