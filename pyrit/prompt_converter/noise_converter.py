import logging
import textwrap
from typing import Optional
import uuid
import pathlib

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptDataType, PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_converter import PromptConverter, ConverterResult
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class NoiseConverter(PromptConverter):
    def __init__(
        self,
        *,
        converter_target: PromptChatTarget,
        noise: Optional[str] = None,
        number_errors: Optional[int] = 5,
        prompt_template: PromptTemplate = None,
    ):
        """
        Injects noise errors into a conversation

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt
            noise (str): The noise to inject. Grammar error, delete random letter, insert random space, etc.
            number_errors (int): The number of errors to inject
            prompt_template (PromptTemplate, optional): The prompt template for the conversion.

        """
        self.converter_target = converter_target

        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "noise_converter.yaml"
            )
        )

        if not noise:
            noise = textwrap.dedent(
                "Grammar error, Delete random letter, insert random symbol, missing white space, "
                "bad auto-correct, or similar"
            )

        self.system_prompt = prompt_template.apply_custom_metaprompt_parameters(
            noise=noise, number_errors=str(number_errors)
        )

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Generates a prompt with noise errors

        Parameters:
            prompt (str): The prompt to convert.
            input_type (PromptDataType, optional): The data type of the input prompt. Defaults to "text".

        Returns:
            ConverterResult: The result of the conversion, including the converted output text and output type.

        Raises:
            ValueError: If the input type is not supported.
        """

        conversation_id = str(uuid.uuid4())

        self.converter_target.set_system_prompt(
            system_prompt=self.system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
        )

        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=prompt,
                    converted_value=prompt,
                    conversation_id=conversation_id,
                    sequence=1,
                    prompt_target_identifier=self.converter_target.get_identifier(),
                    original_value_data_type=input_type,
                    converted_value_data_type=input_type,
                    converter_identifiers=[self.get_identifier()],
                )
            ]
        )

        response = await self.converter_target.send_prompt_async(prompt_request=request)
        return ConverterResult(output_text=response.request_pieces[0].converted_value, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
