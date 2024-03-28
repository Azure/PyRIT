# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import ChatMessage
from pyrit.chat_message_normalizer import ChatMessageNormalizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class ChatMessageNormalizerTokenizerTemplate(ChatMessageNormalizer[str]):

    def __init__(self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    def normalize(self, messages: list[ChatMessage]) -> str:
        """Convert a string of text to the format specified by the chat
        template loaded in a custom Hugging Face tokenizer.
        This method utilizes the `apply_chat_template` method documented at
        https://huggingface.co/docs/transformers/main/en/chat_templating
        """

        messages_list = []

        formatted_messages: str = ""
        for m in messages:
            messages_list.append({"role": m.role, "content": m.content})
        formatted_messages = self.tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted_messages
