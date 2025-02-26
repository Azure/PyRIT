# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import random
import uuid
from typing import List, Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPrompt
from pyrit.models.literals import PromptDataType
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.prompt_converter.fuzzer_converter.fuzzer_converter_base import (
    FuzzerConverter,
)
from pyrit.prompt_converter.prompt_converter import ConverterResult
from pyrit.prompt_target import PromptChatTarget


class FuzzerCrossOverConverter(FuzzerConverter):
    """
    Fuzzer converter that uses multiple prompt templates to generate new prompts.

    Parameters

    converter_target: PromptChatTarget
        Chat target used to perform fuzzing on user prompt

    prompt_template: SeedPrompt, default=None
        Template to be used instead of the default system prompt with instructions for the chat target.

    prompt_templates: List[str], default=None
        List of prompt templates to use in addition to the default template.
    """

    def __init__(
        self,
        *,
        converter_target: PromptChatTarget,
        prompt_template: SeedPrompt = None,
        prompt_templates: Optional[List[str]] = None,
    ):
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "fuzzer_converters" / "crossover_converter.yaml"
            )
        )
        super().__init__(converter_target=converter_target, prompt_template=prompt_template)
        self.prompt_templates = prompt_templates or []
        self.template_label = "TEMPLATE 1"

    def update(self, **kwargs) -> None:
        if "prompt_templates" in kwargs:
            self.prompt_templates = kwargs["prompt_templates"]

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converter to generate versions of prompt with new, prepended sentences.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        if len(self.prompt_templates) == 0:
            raise ValueError(
                "No prompt templates available for crossover. Please provide prompt templates via the update method."
            )

        conversation_id = str(uuid.uuid4())

        self.converter_target.set_system_prompt(
            system_prompt=self.system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
        )

        formatted_prompt = f"===={self.template_label} BEGINS====\n{prompt}\n===={self.template_label} ENDS===="
        formatted_prompt += (
            f"\n====TEMPLATE 2 BEGINS====\n{random.choice(self.prompt_templates)}\n====TEMPLATE 2 ENDS====\n"
        )

        prompt_metadata: dict[str, str | int] = {"response_format": "json"}
        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=formatted_prompt,
                    converted_value=formatted_prompt,
                    conversation_id=conversation_id,
                    sequence=1,
                    prompt_target_identifier=self.converter_target.get_identifier(),
                    original_value_data_type=input_type,
                    converted_value_data_type=input_type,
                    converter_identifiers=[self.get_identifier()],
                    prompt_metadata=prompt_metadata,
                )
            ]
        )

        response = await self.send_prompt_async(request)

        return ConverterResult(output_text=response, output_type="text")
