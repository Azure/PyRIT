# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import re
from pathlib import Path
from typing import Optional

import yaml

from pyrit.identifiers import ConverterIdentifier
from pyrit.models import PromptDataType
from pyrit.prompt_converter.prompt_converter import ConverterResult, PromptConverter


class ColloquialWordswapConverter(PromptConverter):
    """
    Converts text into colloquial Singaporean context.
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    def __init__(self, deterministic: bool = False, wordswap_path: Optional[str] = None) -> None:
        """
        Initialize the converter with optional deterministic mode and custom substitutions.

        Args:
            deterministic (bool): If True, use the first substitution for each wordswap.
                If False, randomly choose a substitution for each wordswap. Defaults to False.
            wordswap_path (Optional[str]): File name of a YAML file in ../../datasets/prompt_converters/colloquial_wordswaps
                directory containing a dictionary of substitutions. Defaults to None.

        Raises:
            FileNotFoundError: If the wordswap YAML file is not found.
            ValueError: If the YAML file is formatted incorrectly or empty.
        """
        # Use custom substitutions if wordswap_path provided, otherwise default to singaporean.yaml
        if wordswap_path:
            file_path = (
                Path(__file__).parent.parent / "datasets" / "prompt_converters" / "colloquial_wordswaps" / wordswap_path
            )
        else:
            file_path = (
                Path(__file__).parent.parent
                / "datasets"
                / "prompt_converters"
                / "colloquial_wordswaps"
                / "singaporean.yaml"
            )

        if not file_path.exists():
            raise FileNotFoundError(f"Colloquial wordswap file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Ensure that wordswap YAML is in the correct format.
        if not isinstance(data, dict):
            raise ValueError("Wordswap YAML must contain a dictionary of word -> list of substitutions")

        self._colloquial_substitutions = data
        self._deterministic = deterministic

    def _build_identifier(self) -> ConverterIdentifier:
        """
        Build identifier with colloquial wordswap parameters.

        Returns:
            ConverterIdentifier: The identifier for this converter.
        """
        return self._create_identifier(
            converter_specific_params={
                "deterministic": self._deterministic,
                "substitution_keys": sorted(self._colloquial_substitutions.keys()),
            }
        )

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert the given prompt by replacing words with colloquial Singaporean terms.

        Args:
            prompt (str): The input text prompt to be converted.
            input_type (PromptDataType): The type of the input prompt. Defaults to "text".

        Returns:
            ConverterResult: The result containing the converted prompt.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        # Tokenize the prompt into words and non-words
        words = re.findall(r"\w+|\S+", prompt)
        converted_prompt = []

        for word in words:
            lower_word = word.lower()
            if lower_word in self._colloquial_substitutions:
                if self._deterministic:
                    # Use the first substitution for deterministic mode
                    converted_prompt.append(self._colloquial_substitutions[lower_word][0])
                else:
                    # Randomly select a substitution for each wordswap
                    converted_prompt.append(random.choice(self._colloquial_substitutions[lower_word]))
            else:
                # If word not in substitutions, keep it as is
                converted_prompt.append(word)

        # Join all words and punctuation with spaces
        final_prompt = " ".join(converted_prompt)

        # Clean up spaces for final prompt
        final_prompt = re.sub(r'\s([?.!,\'"])', r"\1", final_prompt)
        final_prompt = final_prompt.strip()

        return ConverterResult(output_text=final_prompt, output_type="text")
