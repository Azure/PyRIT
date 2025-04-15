# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import PreTrainedTokenizerBase

from pyrit.chat_message_normalizer import ChatMessageNormalizer
from pyrit.models import ChatMessage


class ChatMessageNormalizerTokenizerTemplate(ChatMessageNormalizer[str]):
    """
    This class enables you to apply the chat template stored in a Hugging Face tokenizer
    to a list of chat messages. For more details, see
    https://huggingface.co/docs/transformers/main/en/chat_templating
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """
        Initializes an instance of the ChatMessageNormalizerTokenizerTemplate class.

        Args:
            tokenizer (PreTrainedTokenizerBase): A Hugging Face tokenizer.
        """
        self.tokenizer = tokenizer

    def normalize(self, messages: list[ChatMessage]) -> str:
        """
        Applies the chat template stored in the tokenizer to a list of chat messages.

        Args:
            messages (list[ChatMessage]): A list of ChatMessage objects.

        Returns:
            str: The formatted chat messages.
        """

        messages_list = []

        formatted_messages: str = ""
        for m in messages:
            messages_list.append({"role": m.role, "content": m.content})

        formatted_messages = str(
            self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        return formatted_messages
