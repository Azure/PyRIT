# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptDataType, SeedPrompt
from pyrit.prompt_converter import ConverterResult, LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class MaliciousQuestionGeneratorConverter(LLMGenericTextConverter):
    """
    Generates malicious questions using an LLM.

    An existing ``PromptChatTarget`` is used to perform the conversion (like Azure OpenAI).
    """

    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: SeedPrompt = None):
        """
        Initializes the converter with a specific target and template.

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt.
            prompt_template (SeedPrompt): The seed prompt template to use.
        """

        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "malicious_question_generator_converter.yaml"
            )
        )

        super().__init__(converter_target=converter_target, system_prompt_template=prompt_template)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        # Add the prompt to _prompt_kwargs before calling the base method
        self._prompt_kwargs["prompt"] = prompt
        return await super().convert_async(prompt=prompt, input_type=input_type)

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
