# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid

from pyrit.models import PromptDataType, PromptRequestPiece, PromptRequestResponse, SeedPrompt
from pyrit.prompt_converter import PromptConverter, ConverterResult
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class LLMGenericTextConverter(PromptConverter):
    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: SeedPrompt, **kwargs):
        """
        Generic LLM converter that expects text to be transformed (e.g. no JSON parsing or format)

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt
            prompt_template (SeedPrompt, Optional): The prompt template to set as the system prompt.
            kwargs: Additional parameters for the prompt template.

        """
        self._converter_target = converter_target
        self._prompt_template = prompt_template
        self._prompt_kwargs = kwargs

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert a prompt based on the prompt template

        Parameters:
            prompt (str): The prompt to convert.
            input_type (PromptDataType, Optional): The data type of the input prompt. Defaults to "text".

        Returns:
            ConverterResult: The result of the conversion, including the converted output text and output type.

        Raises:
            ValueError: If the input type is not supported.
        """

        conversation_id = str(uuid.uuid4())

        kwargs = self._prompt_kwargs.copy()

        system_prompt = self._prompt_template.render_template_value(**kwargs)

        self._converter_target.set_system_prompt(
            system_prompt=system_prompt,
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
                    prompt_target_identifier=self._converter_target.get_identifier(),
                    original_value_data_type=input_type,
                    converted_value_data_type=input_type,
                    converter_identifiers=[self.get_identifier()],
                )
            ]
        )

        response = await self._converter_target.send_prompt_async(prompt_request=request)
        return ConverterResult(output_text=response.request_pieces[0].converted_value, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
