# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptDataType, SeedPrompt
from pyrit.prompt_converter import ConverterResult, LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class DenylistConverter(LLMGenericTextConverter):
    """Eliminates forbidden words or phrases in a prompt by replacing them with synonyms."""

    def __init__(
        self,
        *,
        converter_target: PromptChatTarget,
        system_prompt_template: Optional[SeedPrompt] = None,
        denylist: list[str] = [],
    ):
        # set to default strategy if not provided
        system_prompt_template = (
            system_prompt_template
            if system_prompt_template
            else SeedPrompt.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "denylist_converter.yaml"
            )
        )

        super().__init__(
            converter_target=converter_target, system_prompt_template=system_prompt_template, denylist=denylist
        )

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Removes any words or phrases from the input prompt that are in the denylist, replacing them
        with synonomous words.

        Args:
            prompt (str): The prompt to be converted.
        Returns:
            str: The converted prompt without any denied words.
        """

        # check if the prompt contains any words from the  denylist and if so,
        # update the prompt replacing the denied words with synonyms
        denylist = self._prompt_kwargs.get("denylist", [])
        if any(word in prompt for word in denylist):
            return await super().convert_async(prompt=prompt, input_type=input_type)
        logger.info(f"Prompt does not contain any words from the denylist. prompt: {prompt}, denylist: {denylist}")
        return ConverterResult(output_text=prompt, output_type=input_type)
