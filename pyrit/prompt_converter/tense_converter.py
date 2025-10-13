# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPrompt
from pyrit.prompt_converter import LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget
from pyrit.setup.pyrit_default_value import apply_defaults

logger = logging.getLogger(__name__)


class TenseConverter(LLMGenericTextConverter):
    """
    Converts a conversation to a different tense using an LLM.

    An existing ``PromptChatTarget`` is used to perform the conversion (like Azure OpenAI).
    """

    @apply_defaults
    def __init__(
        self, *, converter_target: Optional[PromptChatTarget] = None, tense: str, prompt_template: Optional[SeedPrompt] = None
    ):
        """
        Initializes the converter with the target chat support, tense, and optional prompt template.

        Args:
            converter_target (PromptChatTarget): The target chat support for the conversion which will translate.
            tone (str): The tense the converter should convert the prompt to. E.g. past, present, future.
            prompt_template (SeedPrompt, Optional): The prompt template for the conversion.

        Raises:
            ValueError: If the language is not provided.
        """
        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(pathlib.Path(DATASETS_PATH) / "prompt_converters" / "tense_converter.yaml")
        )

        super().__init__(
            converter_target=converter_target,
            system_prompt_template=prompt_template,
            tense=tense,
        )
