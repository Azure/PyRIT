# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter
from typing import Optional
# Unicode combining characters for Zalgo effect

class ZalgoConverter(PromptConverter):
    def __init__(self, *,  intensity: int = 10, seed: Optional[int] = None) -> None:
        """
        Initializes zalgo converter with optional seed and variable intensity to make
        more or less cursed text
        Args:
            intensity (int): Adds more marks per character the higher the intensity.  Default = 10
            seed (int): uses optional seed to get same output for comparison purposes
        """
        self._intensity = intensity
        self._seed = seed
    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that z̜͓͗͠aͣ̈́ͪ̎͝l̛̜͇͐g̭o̻̦-ī̧̪̯͋f͕̹̟͋ỉ̓ͭͅe̪̳s͎̥̋̚ͅ t̨͔ͫ͌ęx̭̭t͕͕̳
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        
        if self._seed is not None:
            random.seed(self._seed)
        
        mark = [chr(code) for code in range(0x0300, 0x036F + 1)]
        def glitch(char):
            result = char
            for _ in range(random.randint(1, self._intensity)):
                result += random.choice(mark)
            return result
        
        output_text = ''.join(glitch(c) if c.isalnum() else c for c in prompt)
        result = ConverterResult(output_text=output_text, output_type="text")
        return result
    
    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
