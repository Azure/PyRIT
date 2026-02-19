# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Literal, Optional

from pyrit.common.apply_defaults import REQUIRED_VALUE, apply_defaults
from pyrit.common.path import CONVERTER_SEED_PROMPT_PATH
from pyrit.identifiers import ConverterIdentifier
from pyrit.models import SeedPrompt
from pyrit.prompt_converter.llm_generic_text_converter import LLMGenericTextConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)

# Supported obfuscation modes
ObfuscationMode = Literal["academic", "technical", "smiles", "research", "reaction", "combined"]


class ScientificObfuscationConverter(LLMGenericTextConverter):
    """
    Uses an LLM to transform simple or direct prompts into
    scientifically-framed versions using technical terminology, chemical notation,
    or academic phrasing. This can be useful for red-teaming scenarios to test
    whether safety filters can be bypassed through scientific obfuscation.

    Supports multiple modes:
        - ``academic``: Frame as scholarly, homework style questions
        - ``technical``: Use precise technical jargon and nomenclature
        - ``smiles``: Use SMILES chemical notation and IUPAC names
        - ``research``: Frame as legitimate research question
        - ``reaction``: Frame as a step-by-step chemistry mechanism problem
        - ``combined``: Use all techniques together

    """

    @apply_defaults
    def __init__(
        self,
        *,
        converter_target: PromptChatTarget = REQUIRED_VALUE,  # type: ignore[assignment]
        mode: ObfuscationMode = "combined",
        prompt_template: Optional[SeedPrompt] = None,
    ) -> None:
        """
        Initialize the scientific obfuscation converter.

        Args:
            converter_target (PromptChatTarget): The LLM target to perform the conversion.
                Can be omitted if a default has been configured via PyRIT initialization.
            mode (ObfuscationMode): The obfuscation mode to use. Options are:
                - ``academic``: Use academic/scholarly framing
                - ``technical``: Use technical jargon and terminology
                - ``smiles``: Use SMILES notation and chemical nomenclature
                - ``research``: Frame as research/safety study
                - ``reaction``: Frame as a step-by-step chemistry mechanism problem
                - ``combined``: Use all techniques together (default)
            prompt_template (SeedPrompt, Optional): Custom prompt template. If not provided,
                the default scientific_obfuscation_converter.yaml template will be used.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        valid_modes = ("academic", "technical", "smiles", "research", "reaction", "combined")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        # Load default template if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(
                pathlib.Path(CONVERTER_SEED_PROMPT_PATH) / "scientific_obfuscation_converter.yaml"
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
