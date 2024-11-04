# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from pyrit.common.path import DATASETS_PATH
import uuid
from pyrit.models.literals import PromptDataType
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models import SeedPrompt
from pyrit.prompt_converter.fuzzer_converter.fuzzer_converter_base import FuzzerConverter
from pyrit.prompt_converter.prompt_converter import ConverterResult
from pyrit.prompt_target import PromptChatTarget


class FuzzerExpandConverter(FuzzerConverter):
    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: SeedPrompt = None):
        prompt_template = (
            prompt_template
            if prompt_template
            else SeedPrompt.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "fuzzer_converters" / "expand_converter.yaml"
            )
        )
        super().__init__(converter_target=converter_target, prompt_template=prompt_template)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converter to generate versions of prompt with new, prepended sentences.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        conversation_id = str(uuid.uuid4())

        self.converter_target.set_system_prompt(
            system_prompt=self.system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
        )

        formatted_prompt = f"===={self.template_label} BEGINS====\n{prompt}\n===={self.template_label} ENDS===="

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
                )
            ]
        )

        response = await self.send_prompt_async(request)

        return ConverterResult(output_text=response + " " + prompt, output_type="text")
