# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import pathlib
import uuid

from pyrit.common.path import DATASETS_PATH
from pyrit.exceptions import (
    InvalidJsonException,
    pyrit_json_retry,
    remove_markdown_json,
)
from pyrit.models import (
    PromptDataType,
    PromptRequestPiece,
    PromptRequestResponse,
    SeedPrompt,
)
from pyrit.prompt_converter import ConverterResult, PromptConverter
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class PersuasionConverter(PromptConverter):
    """
    Converter to rephrase prompts using a variety of persuasion techniques.

    Based on https://arxiv.org/abs/2401.06373 by Zeng et al.

    Parameters
    ---
    converter_target: PromptChatTarget
        Chat target used to perform rewriting on user prompt

    persuasion_technique:
    {"authority_endorsement", "evidence_based", "expert_endorsement", "logical_appeal", "misrepresentation"}
        Persuasion technique to be used by the converter, determines the system prompt to be used to
        generate new prompts.
        - authority_endorsement: Citing authoritative sources in support of a claim.
        - evidence_based: Using empirical data, statistics, and facts to support a claim or decision.
        - expert_endorsement: Citing domain experts in support of a claim.
        - logical_appeal: Using logic or reasoning to support a claim.
        - misrepresentation: Presenting oneself or an issue in a way that's not genuine or true.
    """

    def __init__(self, *, converter_target: PromptChatTarget, persuasion_technique: str):
        self.converter_target = converter_target

        try:
            prompt_template = SeedPrompt.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "persuasion" / f"{persuasion_technique}.yaml"
            )
        except FileNotFoundError:
            raise ValueError(f"Persuasion technique '{persuasion_technique}' does not exist or is not supported.")
        self.system_prompt = str(prompt_template.value)

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

        response = await self.send_persuasion_prompt_async(request)

        return ConverterResult(output_text=response, output_type="text")

    @pyrit_json_retry
    async def send_persuasion_prompt_async(self, request):
        response = await self.converter_target.send_prompt_async(prompt_request=request)

        response_msg = response.get_value()
        response_msg = remove_markdown_json(response_msg)

        try:
            parsed_response = json.loads(response_msg)
            if "mutated_text" not in parsed_response:
                raise InvalidJsonException(
                    message=f"Invalid JSON encountered; missing 'mutated_text' key: {response_msg}"
                )
            return parsed_response["mutated_text"]

        except json.JSONDecodeError:
            raise InvalidJsonException(message=f"Invalid JSON encountered: {response_msg}")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"
