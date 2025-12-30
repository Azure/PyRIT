# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import List, cast

from pyrit.message_normalizer.message_normalizer import MessageStringNormalizer
from pyrit.models import ALLOWED_CHAT_MESSAGE_ROLES, ChatMessageRole, Message


class ChatMLNormalizer(MessageStringNormalizer):
    """A message normalizer that converts a list of messages to a ChatML string."""

    async def normalize_string_async(self, messages: List[Message]) -> str:
        """
        Convert a list of messages to a ChatML string.

        This is compliant with the ChatML specified in
        https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md

        Args:
            messages: The list of Message objects to normalize.

        Returns:
            The normalized ChatML string.
        """
        final_string = ""
        for message in messages:
            for piece in message.message_pieces:
                content = piece.converted_value or piece.original_value
                final_string += f"<|im_start|>{piece.role}\n{content}<|im_end|>\n"
        return final_string

    @staticmethod
    def from_chatml(content: str) -> List[Message]:
        """
        Convert a ChatML string to a list of messages.

        Args:
            content: The ChatML string to convert.

        Returns:
            The list of Message objects.

        Raises:
            ValueError: If the input content is invalid.
        """
        messages: List[Message] = []
        matches = list(re.finditer(r"<\|im_start\|>(.*?)<\|im_end\|>", content, re.DOTALL | re.MULTILINE))
        if not matches:
            raise ValueError("No chat messages found in the ChatML string")
        for match in matches:
            lines = match.group(1).split("\n")
            role_line = lines[0].strip()
            role_match = re.match(r"(?P<role>\w+)( name=(?P<name>\w+))?", role_line)
            role = role_match.group("role") if role_match else "user"
            if role not in ALLOWED_CHAT_MESSAGE_ROLES:
                raise ValueError(f"Role {role} is not allowed in ChatML")
            message_content = "\n".join(lines[1:]).strip()
            messages.append(Message.from_prompt(prompt=message_content, role=cast(ChatMessageRole, role)))
        return messages
