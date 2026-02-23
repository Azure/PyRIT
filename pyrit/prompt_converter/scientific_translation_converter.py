# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Literal, Optional, get_args

from pyrit.common.apply_defaults import REQUIRED_VALUE, apply_defaults
from pyrit.common.path import CONVERTER_SEED_PROMPT_PATH
from pyrit.identifiers import ConverterIdentifier
from pyrit.models import SeedPrompt
from pyrit.prompt_converter.llm_generic_text_converter import LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


# Supported translation modes
TranslationMode = Literal["academic", "technical", "smiles", "math", "research", "reaction", "combined"]
TRANSLATION_MODES = set(get_args(TranslationMode))


class ScientificTranslationConverter(LLMGenericTextConverter):
    """
    Uses an LLM to transform simple or direct prompts into
    scientifically-framed versions using technical terminology, chemical notation,
    or academic phrasing. This can be useful for red-teaming scenarios to test
    whether safety filters can be bypassed through scientific translation.

    """

    @apply_defaults
    def __init__(
        self,
        *,
        converter_target: PromptChatTarget = REQUIRED_VALUE,  # type: ignore[assignment]
        mode: TranslationMode = "combined",
        prompt_template: Optional[SeedPrompt] = None,
    ) -> None:
        """
        Initialize the scientific translation converter.

        Args:
            converter_target (PromptChatTarget): The LLM target to perform the conversion.
            mode (TranslationMode): The translation mode to use. Options are:
                - ``academic``: Use academic/homework style framing
                - ``technical``: Use technical jargon and terminology
                - ``smiles``: Uses chemical notation
                    eg SMILES [chemical structure using text notation] or IUPAC [the international standard for naming chemicals] notation)
                    ie "2-(acetyloxy)benzoic acid" or "CC(=O)Oc1ccccc1C(=O)O" for aspirin
                - ``research``: Frame as research/safety study or question
                - ``reaction``: Frame as a step-by-step chemistry mechanism problem
                - ``math``: Frame as the answer key to a mathematical problem or equation for a homework/exam setting
                - ``combined``: Use combination of above techniques together (default)
            prompt_template (SeedPrompt, Optional): Custom prompt template. If not provided,
                the default scientific_translation_converter.yaml template will be used.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        if mode not in TRANSLATION_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {TRANSLATION_MODES}")

        # Load default template if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(
                pathlib.Path(CONVERTER_SEED_PROMPT_PATH) / "scientific_translation_converter.yaml"
            )
        )

        super().__init__(
            converter_target=converter_target,
            system_prompt_template=prompt_template,
            mode=mode,
        )
        self._mode = mode

    def _build_identifier(self) -> ConverterIdentifier:
        """
        Build the converter identifier with mode parameter.

        Returns:
            ConverterIdentifier: The identifier for this converter including the mode.
        """
        return self._create_identifier(
            converter_target=self._converter_target,
            converter_specific_params={
                "mode": self._mode,
            },
        )
