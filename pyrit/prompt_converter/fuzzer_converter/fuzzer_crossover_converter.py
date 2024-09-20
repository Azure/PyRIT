import pathlib
import random
import uuid
from pyrit.common.path import DATASETS_PATH
from typing import List, Optional
from pyrit.models.literals import PromptDataType
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.prompt_template import PromptTemplate
from pyrit.prompt_converter.fuzzer_converter.fuzzer_converter_base import FuzzerConverter
from pyrit.prompt_converter.prompt_converter import ConverterResult
from pyrit.prompt_target.prompt_chat_target.prompt_chat_target import PromptChatTarget


class FuzzerCrossOverConverter(FuzzerConverter):
    def __init__(
        self,
        *,
        converter_target: PromptChatTarget,
        prompt_template: PromptTemplate = None,
        prompts: Optional[List[str]] = None,
    ):
        prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "crossover_converter.yaml"
            )
        )
        super().__init__(converter_target=converter_target, prompt_template=prompt_template)
        self.prompts = prompts or []
        self.template_label = "TEMPLATE 1"

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
        formatted_prompt += f"\n====TEMPLATE 2 BEGINS====\n{random.choice(self.prompts)}\n====TEMPLATE 2 ENDS====\n"

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

        return ConverterResult(output_text=response, output_type="text")