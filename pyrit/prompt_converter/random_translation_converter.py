# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
import re
from pathlib import Path
from typing import List, Optional, Union

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptDataType, SeedDataset, SeedPrompt
from pyrit.prompt_converter import ConverterResult, LLMGenericTextConverter
from pyrit.prompt_converter.word_level_converter import WordLevelConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class RandomTranslationConverter(LLMGenericTextConverter, WordLevelConverter):
    """
    Translates each individual word in a prompt to a random language using an LLM.

    An existing ``PromptChatTarget`` is used to perform the translation (like Azure OpenAI).
    """

    # Default language list
    _DEFAULT_LANGUAGES_SEED_PROMPT_PATH = Path(DATASETS_PATH) / "lexicons" / "languages_most_spoken.yaml"

    def __init__(
        self,
        *,
        converter_target: PromptChatTarget,
        system_prompt_template: Optional[SeedPrompt] = None,
        languages: Optional[List[str]] = None,
        indices: Optional[List[int]] = None,
        keywords: Optional[List[str]] = None,
        proportion: Optional[float] = None,
        regex: Optional[Union[str, re.Pattern]] = None,
    ):
        """
        Initializes the converter with a target, an optional system prompt template, and a denylist.

        Args:
            converter_target (PromptChatTarget): The target for the prompt conversion.
            system_prompt_template (Optional[SeedPrompt]): The system prompt template to use for the conversion.
                If not provided, a default template will be used.
            languages (Optional[List[str]]): The list of available languages to use for translation.
            indices (Optional[List[int]]): Specific indices of words to convert.
            keywords (Optional[List[str]]): Keywords to select words for conversion.
            proportion (Optional[float]): Proportion of randomly selected words to convert [0.0-1.0].
            regex (Optional[Union[str, re.Pattern]]): Regex pattern to match words for conversion.
        """
        # set to default strategy if not provided
        system_prompt_template = (
            system_prompt_template
            if system_prompt_template
            else SeedPrompt.from_yaml_file(
                Path(DATASETS_PATH) / "prompt_converters" / "random_translation_converter.yaml"
            )
        )

        LLMGenericTextConverter.__init__(
            self,
            converter_target=converter_target,
            system_prompt_template=system_prompt_template,
        )

        WordLevelConverter.__init__(
            self, indices=indices, keywords=keywords, proportion=proportion, regex=regex, word_split_separator=None
        )

        if not languages:
            default_languages = SeedDataset.from_yaml_file(
                RandomTranslationConverter._DEFAULT_LANGUAGES_SEED_PROMPT_PATH
            )
            self.languages = [prompt.value for prompt in default_languages.prompts]
        else:
            self.languages = languages

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts the given prompt into the target format supported by the converter.

        Args:
            prompt (str): The prompt to be converted.
            input_type (PromptDataType): The type of input data.

        Returns:
            ConverterResult: The result containing the converted output and its type.
        """
        words = prompt.split()
        selected_indices = self._select_word_indices(words=words)

        language_word_pairs = [f"NOOP: {word}" for word in words]

        # Translate only selected words
        for idx in selected_indices:
            language_word_pairs[idx] = await self.convert_word_async(words[idx])

        llm_prompt = ""
        for pair in language_word_pairs:
            llm_prompt += f"{pair}\n"

        return await LLMGenericTextConverter.convert_async(self, prompt=llm_prompt, input_type="text")

    async def convert_word_async(self, word: str) -> str:
        """
        Converts a single word into the target format supported by the converter.

        Args:
            word (str): The word to be converted.

        Returns:
            str: The converted word.
        """
        language = random.choice(self.languages)
        return f"{language}: {word}"
