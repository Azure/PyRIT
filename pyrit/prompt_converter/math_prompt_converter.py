# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.prompt_converter import LLMGenericTextConverter
from pyrit.models import PromptTemplate
from pyrit.prompt_target import PromptChatTarget

from pyrit.common.path import DATASETS_PATH
import pathlib

logger = logging.getLogger(__name__)


class MathPromptConverter(LLMGenericTextConverter):
    """
    A PromptConverter that converts natural language instructions into symbolic mathematics problems 
    using an LLM via an existing PromptTarget (like Azure OpenAI or other supported backends).
    """

    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: PromptTemplate = None):
        """
        Initializes the converter with a specific target and template.

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt.
            prompt_template (PromptTemplate): The prompt template to use.
        """

        # Load the template from the YAML file or use a default template if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "math_prompt_converter.yaml"
            )
        )

        super().__init__(converter_target=converter_target, prompt_template=prompt_template)

    async def convert_async(self, *, prompt: str) -> str:
        """
        Convert a prompt into a mathematical problem format.

        Parameters:
            prompt (str): The prompt to convert.

        Returns:
            str: The result of the conversion, including the mathematical representation and real-world example.
        """
        logger.info(f"Converting prompt: {prompt}")
        return await super().convert_async(prompt=prompt)
