# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
import textwrap
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPrompt
from pyrit.prompt_converter import LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class NoiseConverter(LLMGenericTextConverter):
    """
    Injects noise errors into a conversation using an LLM.

    An existing ``PromptChatTarget`` is used to perform the conversion (like Azure OpenAI).
    """

    def __init__(
        self,
        *,
        converter_target: PromptChatTarget,
        noise: Optional[str] = None,
        number_errors: int = 5,
        prompt_template: Optional[SeedPrompt] = None,
    ):
        """
        Initializes the converter with the specified parameters.

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt.
            noise (str): The noise to inject. Grammar error, delete random letter, insert random space, etc.
            number_errors (int): The number of errors to inject.
            prompt_template (SeedPrompt, Optional): The prompt template for the conversion.
        """
        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(pathlib.Path(DATASETS_PATH) / "prompt_converters" / "noise_converter.yaml")
        )

        if not noise:
            noise = textwrap.dedent(
                "Grammar error, Delete random letter, insert random symbol, missing white space, "
                "bad auto-correct, or similar"
            )

        super().__init__(
            converter_target=converter_target,
            system_prompt_template=prompt_template,
            noise=noise,
            number_errors=str(number_errors),
        )
