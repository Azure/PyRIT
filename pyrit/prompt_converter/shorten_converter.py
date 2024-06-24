# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import uuid
import pathlib

from pyrit.models import PromptDataType
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_converter import PromptConverter, ConverterResult
from pyrit.models import PromptTemplate
from pyrit.common.path import DATASETS_PATH
from pyrit.prompt_target import PromptChatTarget
from pyrit.exceptions.exception_classes import (
    InvalidJsonException,
    pyrit_json_retry,
    remove_markdown_json,
)

logger = logging.getLogger(__name__)


class ShortenConverter(PromptConverter):
    """
    Converter to shorten or condense prompts.

    Adapted from GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts.

        Link: https://arxiv.org/pdf/2309.10253

        Author: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

        GitHub: https://github.com/sherdencooper/GPTFuzz/tree/master

    Parameters
    ---
    converter_target: PromptChatTarget
        Chat target used to perform condensing on user prompt

    prompt_template: PromptTemplate, default=None
        Template to be used instead of the default system prompt with instructions for the chat target.
    """

    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: PromptTemplate = None):
        self.converter_target = converter_target

        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "shorten_converter.yaml"
            )
        )

        self.system_prompt = prompt_template.template

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converter to generate versions of prompt that is condensed.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        conversation_id = str(uuid.uuid4())

        self.converter_target.set_system_prompt(
            system_prompt=self.system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
        )

        formatted_prompt = f"====TEMPLATE BEGINS====\n{prompt}\n====TEMPLATE ENDS===="

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

        response = await self.send_shorten_prompt_async(request)

        return ConverterResult(output_text=response, output_type="text")

    @pyrit_json_retry
    async def send_shorten_prompt_async(self, request):
        response = await self.converter_target.send_prompt_async(prompt_request=request)

        response_msg = response.request_pieces[0].converted_value
        response_msg = remove_markdown_json(response_msg)

        try:
            parsed_response = json.loads(response_msg)
            if "output" not in parsed_response:
                raise InvalidJsonException(message=f"Invalid JSON encountered; missing 'output' key: {response_msg}")
            return parsed_response["output"]

        except json.JSONDecodeError:
            raise InvalidJsonException(message=f"Invalid JSON encountered: {response_msg}")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
