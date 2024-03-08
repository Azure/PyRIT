# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from pyrit.models import ChatMessage, ChatMessageRole, ALLOWED_CHAT_MESSAGE_ROLES
from pyrit.chat_message_normalizer import ChatMessageNormalizer
from typing import cast


class ChatMessageNormalizerChatML(ChatMessageNormalizer[str]):

    def normalize(self, messages: list[ChatMessage]) -> str:
        """Convert a string of text to a ChatML string.
        This is compliant with the ChatML specified in
        https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md
        """
        final_string: str = ""
        final_string = ""
        for m in messages:
            final_string += f"<|im_start|>{m.role}{f' name={m.name}' if m.name else ''}\n{m.content}<|im_end|>\n"
        return final_string

    @staticmethod
    def from_chatml(content: str) -> list[ChatMessage]:
        """Convert a chatML string to a list of chat messages"""
        messages: list[ChatMessage] = []
        matches = list(re.finditer(r"<\|im_start\|>(.*?)<\|im_end\|>", content, re.DOTALL | re.MULTILINE))
        if not matches:
            raise ValueError("No chat messages found in the chatML string")
        for match in matches:
            lines = match.group(1).split("\n")
            role_line = lines[0].strip()
            role_match = re.match(r"(?P<role>\w+)( name=(?P<name>\w+))?", role_line)
            name = role_match.group("name") if role_match else None
            role = role_match.group("role")
            if role not in ALLOWED_CHAT_MESSAGE_ROLES:
                raise ValueError(f"Role {role} is not allowed in chatML")
            content = lines[1].strip()
            messages.append(ChatMessage(role=cast(ChatMessageRole, role), content=content, name=name))
        return messages
