# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class RepeatTokenConverter(PromptConverter):
    """
    Repeat a specified token a specified number of times in addition to a given prompt.
    Based on:
    https://dropbox.tech/machine-learning/bye-bye-bye-evolution-of-repeated-token-attacks-on-chatgpt-models

    Parameters
    ---
    token_to_repeat: string, default=None
        The string to be repeated

    times_to_repeat: int, default=None
        The number of times the string will be repeated

    token_insert_mode: {"split", "prepend", "append", "repeat"}, default="prepend"
        Method to insert repeated tokens:

        If "split" prompt text will be split on the first occurance of (.?!) punctuation,
        and repeated tokens will be inserted at location of split.

        If "prepend" repeated tokens will be inserted before the prompt text.

        If "append" repeated tokens will be inserted after the prompt text.

        If "repeat" prompt text will be ignored and result will only be repeated tokens.
    """

    def __init__(
        self, *, token_to_repeat: str = None, times_to_repeat: int = None, token_insert_mode: str = "split"
    ) -> None:
        self.token_to_repeat = " " + token_to_repeat.strip()
        self.times_to_repeat = times_to_repeat
        match token_insert_mode:
            case "split":
                # function to split prompt on first punctuation (.?! only), preserve punctuation, 2 parts max.
                def insert(text: str) -> list:
                    parts = re.split(r"(\?|\.|\!)", text, 1)
                    if len(parts) == 3:  # if split mode with no punctuation
                        return [parts[0] + parts[1], parts[2]]
                    return ["", text]

                self.insert = insert
            case "prepend":

                def insert(text: str) -> list:
                    return ["", text]

                self.insert = insert
            case "append":

                def insert(text: str) -> list:
                    return [text, ""]

                self.insert = insert
            case "repeat":

                def insert(text: str) -> list:
                    return ["", ""]

                self.insert = insert
            case _:
                raise ValueError('Invalid insert mode. Must be one of "split", "prepend", "append", or "repeat".')

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converter to insert repeated tokens into the prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        prompt_parts = self.insert(prompt)

        return ConverterResult(
            output_text=f"{prompt_parts[0]}{self.token_to_repeat * self.times_to_repeat}{prompt_parts[1]}",
            output_type="text",
        )

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
