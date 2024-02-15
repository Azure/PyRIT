# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import ChatMessage


def squash_system_message(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Combines the system message into the first user request.

    Args:
        messages: The list of chat messages.

    Returns:
        The list of chat messages with squashed system messages.
    """
    if not messages:
        raise ValueError("ChatMessage list cannot be empty")

    if messages[0].role == "system":
        if len(messages) == 1:
            return [ChatMessage(role="user", content=messages[0].content)]

        first_user_message = combine_system_user_message(messages[0], messages[1])
        return [first_user_message] + messages[2:]

    return messages


def combine_system_user_message(system_message: ChatMessage, user_message: ChatMessage) -> str:
    """Combines the system message with the user message.

    Args:
        system_message (str): The system message.
        user_message (str): The user message.

    Returns:
        str: The combined message.
    """
    content = f"### Instructions ###\n\n{system_message.content}\n\n######\n\n{user_message.content}"
    return ChatMessage(role="user", content=content)