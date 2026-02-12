# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
import uuid
from textwrap import dedent
from typing import List, Optional

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pyrit.common.apply_defaults import REQUIRED_VALUE, apply_defaults
from pyrit.common.path import CONVERTER_SEED_PROMPT_PATH
from pyrit.models import (
    Message,
    MessagePiece,
    PromptDataType,
    SeedPrompt,
)
from pyrit.prompt_converter.prompt_converter import ConverterResult, PromptConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class MultiLanguageTranslationConverter(PromptConverter):
    """
    Translates a prompt into multiple languages by splitting it into segments.

    The prompt is split into word-boundary segments equal to the number of
    requested languages. Each segment is translated into the corresponding
    language and then the translated segments are reassembled into a single
    prompt. If there are more languages than words, only as many languages as
    there are words are used.

    Example:
        prompt = "Hello how are you"
        languages = ["french", "spanish", "italian"]
        segments = ["Hello", "how are", "you"]
        result = "<Hello in French> <how are in Spanish> <you in Italian>"
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    @apply_defaults
    def __init__(
        self,
        *,
        converter_target: PromptChatTarget = REQUIRED_VALUE,  # type: ignore[assignment]
        languages: List[str],
        prompt_template: Optional[SeedPrompt] = None,
        max_retries: int = 3,
        max_wait_time_in_seconds: int = 60,
    ):
        """
        Initialize the converter with the target chat support, languages, and optional prompt template.

        Args:
            converter_target (PromptChatTarget): The target chat support for the conversion which will translate.
                Can be omitted if a default has been configured via PyRIT initialization.
            languages (List[str]): The list of languages to translate segments into.
                E.g. ["French", "Spanish", "Italian"].
            prompt_template (SeedPrompt, Optional): The prompt template for the conversion.
            max_retries (int): Maximum number of retries for each translation call.
            max_wait_time_in_seconds (int): Maximum wait time in seconds between retries.

        Raises:
            ValueError: If converter_target is not provided and no default has been configured.
            ValueError: If languages is empty or not provided.
        """
        self.converter_target = converter_target

        self._max_retries = max_retries
        self._max_wait_time_in_seconds = max_wait_time_in_seconds

        self._prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(
                pathlib.Path(CONVERTER_SEED_PROMPT_PATH) / "translation_converter.yaml"
            )
        )

        if not languages:
            raise ValueError("Languages list must be provided and non-empty for multi-language translation")

        self.languages = [lang.lower() for lang in languages]

    @staticmethod
    def _split_prompt_into_segments(prompt: str, num_segments: int) -> List[str]:
        """
        Split a prompt into segments at word boundaries.

        The prompt is split into ``num_segments`` parts as evenly as possible.
        If there are fewer words than requested segments, each word becomes its
        own segment and the effective number of segments is reduced.

        Args:
            prompt: The text prompt to split.
            num_segments: The desired number of segments.

        Returns:
            A list of string segments.
        """
        words = prompt.split()
        if not words:
            return [prompt] if prompt else []

        # Cap segments to the number of words
        num_segments = min(num_segments, len(words))
        if num_segments <= 0:
            return [prompt]

        # Distribute words as evenly as possible across segments
        base_size, remainder = divmod(len(words), num_segments)
        segments: List[str] = []
        idx = 0
        for i in range(num_segments):
            size = base_size + (1 if i < remainder else 0)
            segments.append(" ".join(words[idx : idx + size]))
            idx += size

        return segments

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert the given prompt by splitting it and translating each segment into a different language.

        Args:
            prompt (str): The prompt to be converted.
            input_type (PromptDataType): The type of input data.

        Returns:
            ConverterResult: The result containing the multi-language translated prompt.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        segments = self._split_prompt_into_segments(prompt, len(self.languages))
        translated_segments: List[str] = []

        for i, segment in enumerate(segments):
            language = self.languages[i]

            system_prompt = self._prompt_template.render_template_value(languages=language)
            conversation_id = str(uuid.uuid4())
            self.converter_target.set_system_prompt(system_prompt=system_prompt, conversation_id=conversation_id)

            formatted_prompt = dedent(
                f"Translate the following to {language} between the begin and end tags:"
                "=== begin ===\n"
                f"{segment}\n"
                "=== end ===\n"
            )

            logger.debug(f"Translating segment '{segment}' to {language}")

            request = Message(
                [
                    MessagePiece(
                        role="user",
                        original_value=segment,
                        converted_value=formatted_prompt,
                        conversation_id=conversation_id,
                        sequence=1,
                        prompt_target_identifier=self.converter_target.get_identifier(),
                        original_value_data_type=input_type,
                        converted_value_data_type=input_type,
                        converter_identifiers=[self.get_identifier()],
                    )
                ]
            )

            translation = await self._send_translation_prompt_async(request)
            translated_segments.append(translation)

        combined = " ".join(translated_segments)
        logger.info(
            "Multi-language translation complete: %d segments across languages %s",
            len(translated_segments),
            self.languages[: len(segments)],
        )
        return ConverterResult(output_text=combined, output_type="text")

    async def _send_translation_prompt_async(self, request: Message) -> str:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=self._max_wait_time_in_seconds),
            retry=retry_if_exception_type(Exception),
        ):
            with attempt:
                logger.debug(f"Attempt {attempt.retry_state.attempt_number} for translation")
                response = await self.converter_target.send_prompt_async(message=request)
                response_msg = response[0].get_value()
                return response_msg.strip()

        raise Exception(f"Failed to translate after {self._max_retries} attempts")
