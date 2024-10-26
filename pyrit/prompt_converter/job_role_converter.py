# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid

from pyrit.prompt_converter import ConverterResult, LLMGenericTextConverter
from pyrit.models import PromptTemplate, PromptDataType, PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_target import PromptChatTarget

from pyrit.common.path import DATASETS_PATH
import pathlib

logger = logging.getLogger(__name__)


class JobRoleGenerator(LLMGenericTextConverter):
    """
    A PromptConverter that adds demographic groups to the job role.
    """

    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: PromptTemplate = None):
        """
        Initializes the converter with a specific target and template.

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt.
            prompt_template (PromptTemplate): The prompt template to use.
        """

        # Set to default strategy if not provided
        self.prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "job_role_converter.yaml"
            )
        )

        super().__init__(converter_target=converter_target, prompt_template=self.prompt_template)

    def input_supported(self, input_type: str) -> bool:
        """
        Check if the input type is supported by the converter.

        Args:
            input_type (str): The input type to check.

        Returns:
            bool: True if the input type is supported, False otherwise.
        """
        return input_type == "text"
    
    async def convert_async(self, *, job: str, demographic: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert a job role into a demographic group.

        Parameters:
            job (str): The job role to convert.
            demographic (str): The demographic group to add to the job role.

        Returns:
            ConverterResult: The result of the conversion, including the job role with demographic group.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        
        conversation_id = str(uuid.uuid4())

        # Set placeholder parameters
        kwargs = self._prompt_kwargs.copy()
        kwargs["job"] = job
        kwargs["demographic"] = demographic

        system_prompt = self._prompt_template.apply_custom_metaprompt_parameters(**kwargs)

        self._converter_target.set_system_prompt(
            system_prompt=system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
        )

        prompt = f"{job} ({demographic})"
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