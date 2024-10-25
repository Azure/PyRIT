# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.prompt_converter import LLMGenericTextConverter
from pyrit.models import SeedPromptTemplate
from pyrit.prompt_target import PromptChatTarget

from pyrit.common.path import DATASETS_PATH
import pathlib

logger = logging.getLogger(__name__)


class MaliciousQuestionGeneratorConverter(LLMGenericTextConverter):
    """
    A PromptConverter that generates malicious questions using an LLM via an existing PromptTarget (like Azure OpenAI).
    """

    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: SeedPromptTemplate = None):
        """
        Initializes the converter with a specific target and template.

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt.
            prompt_template (SeedPromptTemplate): The seed prompt template to use.
        """

        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "malicious_question_generator_converter.yaml"
            )
        )

        super().__init__(converter_target=converter_target, prompt_template=prompt_template)
