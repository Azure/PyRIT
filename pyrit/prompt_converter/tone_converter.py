import logging
import uuid
import pathlib

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptDataType, PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_converter import PromptConverter, ConverterResult
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class ToneConverter(PromptConverter):
    def __init__(self, *, converter_target: PromptChatTarget, tone: str, prompt_template: PromptTemplate = None):
        """
        Converts a conversation to a different tone

        Args:
            converter_target (PromptChatTarget): The target chat support for the conversion which will translate
            tone (str): The tone for the conversation. E.g. upset, sarcastic, indifferent, etc.
            prompt_template (PromptTemplate, optional): The prompt template for the conversion.

        Raises:
            ValueError: If the language is not provided.
        """
        self.converter_target = converter_target

        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "tone_converter.yaml"
            )
        )

        self.system_prompt = prompt_template.apply_custom_metaprompt_parameters(tone=tone)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converts a conversation to a different tone using the converter target.

        Parameters:
            prompt (str): The input prompt to convert.
            input_type (PromptDataType, optional): The data type of the input prompt. Defaults to "text".

        Returns:
            ConverterResult: The result of the conversion, including the output text and output type.
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
