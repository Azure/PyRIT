# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPrompt
from pyrit.prompt_converter import LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class ToneConverter(LLMGenericTextConverter):
    def __init__(self, *, converter_target: PromptChatTarget, tone: str, prompt_template: SeedPrompt = None):
        """
        Converts a conversation to a different tone

        Args:
            converter_target (PromptChatTarget): The target chat support for the conversion which will translate
            tone (str): The tone for the conversation. E.g. upset, sarcastic, indifferent, etc.
            prompt_template (SeedPrompt, Optional): The prompt template for the conversion.

        Raises:
            ValueError: If the language is not provided.
        """
        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(pathlib.Path(DATASETS_PATH) / "prompt_converters" / "tone_converter.yaml")
        )

        super().__init__(
            converter_target=converter_target,
            prompt_template=prompt_template,
            tone=tone,
        )
