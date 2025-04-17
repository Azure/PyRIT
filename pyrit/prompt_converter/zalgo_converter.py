# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import Optional
from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

# Unicode combining characters for Zalgo effect (U+0300–U+036F)
ZALGO_MARKS = [chr(code) for code in range(0x0300, 0x036F + 1)]
# Setting a max intensity so people don't do anything insane
MAX_INTENSITY = 100
class ZalgoConverter(PromptConverter):
    def __init__(self, *, intensity: int = 10, seed: Optional[int] = None) -> None:
        """
        Initializes the Zalgo converter.

        Args:
            intensity (int): Number of combining marks per character (higher = more cursed). Default is 10.
            seed (Optional[int]): Optional seed for reproducible output.
        """
        self._intensity = min(intensity, MAX_INTENSITY)
        self._seed = seed

    async def convert_async(
        self, *, prompt: str, input_type: PromptDataType = "text"
    ) -> ConverterResult:
        """
        Converts text into cursed Zalgo text using combining Unicode marks.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        if self._intensity <= 0:
            output_text = prompt
        else:
            if self._seed is not None:
                random.seed(self._seed)

            def glitch(char: str) -> str:
                return char + ''.join(random.choice(ZALGO_MARKS) for _ in range(random.randint(1, self._intensity)))
            output_text = ''.join(glitch(c) if c.isalnum() else c for c in prompt)
        return ConverterResult(output_text=output_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"

