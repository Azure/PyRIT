# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional
import pathlib

from pyrit.models import PromptTemplate, PromptDataType
from pyrit.prompt_converter import LLMGenericTextConverter, ConverterResult
from pyrit.prompt_target import PromptChatTarget

from pyrit.common.path import DATASETS_PATH


logger = logging.getLogger(__name__)


class JobRoleGenerator(LLMGenericTextConverter):
    """
    A PromptConverter that adds demographic groups to the job role.
    """

    def __init__(
        self, *, converter_target: PromptChatTarget, job: Optional[str], prompt_template: PromptTemplate = None
    ):
        """
        Initializes the converter with a specific target and template.

        Args:
            converter_target (PromptChatTarget): The endpoint that converts the prompt.
            job (str): The job role to append demographic groups to.
            prompt_template (PromptTemplate): The prompt template to use.
        """
        # Set to default strategy if not provided
        self._prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "job_role_converter.yaml"
            )
        )

        super().__init__(converter_target=converter_target, prompt_template=self._prompt_template, job=job)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert a job role into a demographic group.

        Parameters:
            prompt (str): The demographic group to append.
            input_type (PromptDataType): The type of input to convert.

        Returns:
            ConverterResult: The result of the conversion, including the job role with demographic group.

        Raises:
            ValueError: If the input type is not supported.
        """
        return await super().convert_async(prompt=prompt, input_type=input_type)
