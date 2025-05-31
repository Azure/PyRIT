# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import textwrap

import termcolor

from pyrit.models import ChatMessage


def print_chat_messages_with_color(
    messages: list[ChatMessage],
    max_content_character_width: int = 80,
    left_padding_width: int = 20,
    custom_colors: dict[str, str] = None,
) -> None:
    """Print chat messages with color to console.

    Args:
        messages: List of chat messages.
        max_content_character_width: Maximum character width for the content.
        left_padding_width: Maximum character width for the left padding.
        custom_colors: Custom colors for the roles, in the format {"ROLE": "COLOR"}.
            If None, default colors will be used.

    Returns:
        None
    """
    role_to_color: dict[str, str] = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
    }
    if custom_colors:
        role_to_color.update(custom_colors)

    for message in messages:
        output_message = ""
        if message.role == "user":
            output_message = textwrap.fill(
                text=message.content,
                width=max_content_character_width - left_padding_width,
            )
        elif message.role == "system":
            prefix = f"{'SYSTEM INSTRUCTIONS'.center(80, '-')}"
            postfix = f"{'END OF SYSTEM INSTRUCTIONS'.center(80, '-')}"
            output_message = prefix + "\n" + message.content + "\n" + postfix
        else:
            # Threat all non-user messages as assistant messages.
            left_padding = " " * left_padding_width
            output_message = textwrap.fill(
                text=message.content,
                width=max_content_character_width,
                initial_indent=left_padding,
                subsequent_indent=left_padding,
            )
        print("Message with role: " + message.role)
        termcolor.cprint(output_message, color=role_to_color[message.role])  # type: ignore[arg-type]
