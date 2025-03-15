# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.chat_message_normalizer import ChatMessageNormalizer
from pyrit.models import ChatMessage, ChatMessageRole


class GenericSystemSquash(ChatMessageNormalizer[list[ChatMessage]]):
    def normalize(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Returns the first system message combined with the first user message.

        The format of the result uses generic instruction tags.

        Args:
            messages (list[ChatMessage]): The list of messages to normalize.

        Returns:
            list[ChatMessage]: The normalized list of messages.
        """
        normalized_messages = ChatMessageNormalizer.squash_system_message(
            messages=messages, squash_function=GenericSystemSquash.combine_system_user_message
        )
        return normalized_messages

    @staticmethod
    def combine_system_user_message(
        system_message: ChatMessage, user_message: ChatMessage, msg_type: ChatMessageRole = "user"
    ) -> ChatMessage:
        """Combines the system message with the user message.

        Args:
            system_message (str): The system message.
            user_message (str): The user message.
            msg_type (ChatMessageRole, optional): The message type. Defaults to "user".

        Returns:
            ChatMessage: The combined message.
        """
        content = f"### Instructions ###\n\n{system_message.content}\n\n######\n\n{user_message.content}"
        return ChatMessage(role=msg_type, content=content)
