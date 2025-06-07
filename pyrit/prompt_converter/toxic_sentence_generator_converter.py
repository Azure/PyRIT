# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Toxic Sentence Generator Converter module for generating potentially harmful content
to test AI safety mechanisms.
"""

import logging
import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptDataType, SeedPrompt
from pyrit.prompt_converter import ConverterResult, LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class ToxicSentenceGeneratorConverter(LLMGenericTextConverter):
    """
    A PromptConverter that generates toxic sentence starters using an LLM via an
    existing PromptTarget.

    Based on Project Moonshot's attack module that generates toxic sentences to test LLM
    safety guardrails:
    https://github.com/aiverify-foundation/moonshot-data/blob/main/attack-modules/toxic_sentence_generator.py
    """

    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: Optional[SeedPrompt] = None):
        """
        Initializes the converter with a specific target and template.

        Parameters:
            converter_target (PromptChatTarget): The endpoint that converts the prompt.
            prompt_template (SeedPrompt): The seed prompt template to use. If not provided,
                                          defaults to the toxic_sentence_generator.yaml.
        """

        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "toxic_sentence_generator.yaml"
            )
        )

        super().__init__(converter_target=converter_target, system_prompt_template=prompt_template)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts a seed prompt into a toxic sentence starter.

        Parameters:
            prompt (str): The prompt to convert.
            input_type (PromptDataType, Optional): The data type of the input prompt.
                Defaults to "text".

        Returns:
            ConverterResult: The result of the conversion, containing the toxic sentence
                starter.
        """
        # Add the prompt to _prompt_kwargs before calling the base method
        self._prompt_kwargs["prompt"] = prompt
        return await super().convert_async(prompt=prompt, input_type=input_type)

    def input_supported(self, input_type: PromptDataType) -> bool:
        """
        Checks if the input type is supported by this converter.

        Parameters:
            input_type (PromptDataType): The data type to check.

        Returns:
            bool: True if the input type is supported, False otherwise.
        """
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        """
        Checks if the output type is supported by this converter.

        Parameters:
            output_type (PromptDataType): The data type to check.

        Returns:
            bool: True if the output type is supported, False otherwise.
        """
        return output_type == "text"
