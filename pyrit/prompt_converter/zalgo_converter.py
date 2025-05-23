# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
from typing import Optional

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

# Unicode combining characters for Zalgo effect (U+0300–U+036F)
ZALGO_MARKS = [chr(code) for code in range(0x0300, 0x036F + 1)]
# Setting a max intensity so people don't do anything unreasonable
MAX_INTENSITY = 100
logger = logging.getLogger(__name__)


class ZalgoConverter(PromptConverter):
    def __init__(self, *, intensity: int = 10, seed: Optional[int] = None) -> None:
        """
        Initializes the Zalgo converter.

        Args:
            intensity (int): Number of combining marks per character (higher = more cursed). Default is 10.
            seed (Optional[int]): Optional seed for reproducible output.
        """
        self._intensity = self._normalize_intensity(intensity)
        self._seed = seed

    def _normalize_intensity(self, intensity: int) -> int:
        try:
            intensity = int(intensity)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid intensity value: {intensity!r} (must be an integer)")
        normalized_intensity = max(0, min(intensity, MAX_INTENSITY))
        if intensity != normalized_intensity:
            logger.warning(
                f"ZalgoConverter supports intensity between 0 and {MAX_INTENSITY}, "
                f"but received a value of {intensity}. Normalizing to {normalized_intensity}."
            )
        return normalized_intensity

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts text into cursed Zalgo text using combining Unicode marks.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        def glitch(char: str) -> str:
            return char + "".join(random.choice(ZALGO_MARKS) for _ in range(random.randint(1, self._intensity)))

        if self._intensity <= 0:
            output_text = prompt
        else:
            if self._seed is not None:
                random.seed(self._seed)
            output_text = "".join(glitch(c) if c.isalnum() else c for c in prompt)
        return ConverterResult(output_text=output_text, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
