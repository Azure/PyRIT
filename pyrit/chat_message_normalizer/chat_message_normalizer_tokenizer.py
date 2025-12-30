# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from transformers import PreTrainedTokenizerBase

from pyrit.message_normalizer import MessageNormalizer
from pyrit.models import Message


class TokenizerTemplateNormalizer(MessageNormalizer[str]):
    """
    Enable application of the chat template stored in a Hugging Face tokenizer
    to a list of messages. For more details, see
    https://huggingface.co/docs/transformers/main/en/chat_templating.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Initialize an instance of the TokenizerTemplateNormalizer class.

        Args:
            tokenizer: A Hugging Face tokenizer with a chat template.
        """
        self.tokenizer = tokenizer

    def normalize(self, messages: List[Message]) -> str:
        """
        Apply the chat template stored in the tokenizer to a list of messages.

        Args:
            messages: A list of Message objects.

        Returns:
            The formatted chat messages as a string.
        """
        messages_list = []

        for message in messages:
            for piece in message.message_pieces:
                content = piece.converted_value or piece.original_value
                messages_list.append({"role": piece.role, "content": content})

        formatted_messages = str(
            self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        return formatted_messages


# Backward compatibility alias (deprecated)
ChatMessageNormalizerTokenizerTemplate = TokenizerTemplateNormalizer
