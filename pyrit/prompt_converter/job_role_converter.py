# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.prompt_converter import LLMGenericTextConverter
from pyrit.models import PromptTemplate
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

    def manual_attack_demographics(self, prompt) -> list:
        """
        Source: https://github.com/aiverify-foundation/moonshot-data

        This method prepares prompts for a target Language Learning Model (LLM) using the provided prompt.
        These prompts are designed to generate demographic groups for job roles,
        and the attack can substituted inside the given prompt template and sent to the target LLM.
        """
        result_list = []

        gender = ["Male", "Female"]
        race = ["Malay", "Chinese", "Indian", "White", "Black", "African-American"]

        for g in gender:
            result_list.append(f"{prompt} ({g})")

        for r in race:
            result_list.append(f"{prompt} ({r})")

        return result_list