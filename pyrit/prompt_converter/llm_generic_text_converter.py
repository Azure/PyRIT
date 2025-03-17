# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from typing import Optional

from pyrit.models import (
    PromptDataType,
    PromptRequestPiece,
    PromptRequestResponse,
    SeedPrompt,
)
from pyrit.prompt_converter import ConverterResult, PromptConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class LLMGenericTextConverter(PromptConverter):
    def __init__(
        self,
        *,
        converter_target: PromptChatTarget,
        system_prompt_template: Optional[SeedPrompt] = None,
        user_prompt_template_with_objective: Optional[SeedPrompt] = None,
        **kwargs,
    ):
        """
        Generic LLM converter that expects text to be transformed (e.g. no JSON parsing or format)

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt
            system_prompt_template (SeedPrompt, Optional): The prompt template to set as the system prompt.
            user_prompt_template_with_objective (SeedPrompt, Optional): The prompt template to set as the user prompt.
                expects
            kwargs: Additional parameters for the prompt template.

        """
        self._converter_target = converter_target
        self._system_prompt_template = system_prompt_template
        self._prompt_kwargs = kwargs

        if user_prompt_template_with_objective and "objective" not in user_prompt_template_with_objective.parameters:
            raise ValueError("user_prompt_template_with_objective must contain the 'objective' parameter")

        self._user_prompt_template_with_objective = user_prompt_template_with_objective

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

        if self._system_prompt_template:

            system_prompt = self._system_prompt_template.render_template_value(**kwargs)

            self._converter_target.set_system_prompt(
                system_prompt=system_prompt,
                conversation_id=conversation_id,
                orchestrator_identifier=None,
            )

        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        if self._user_prompt_template_with_objective:
            prompt = self._user_prompt_template_with_objective.render_template_value(objective=prompt)

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=prompt,
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
        return ConverterResult(output_text=response.get_value(), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
