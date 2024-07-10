# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from tests.mocks import MockPromptTarget
from unittest.mock import AsyncMock, patch

from pyrit.common.constants import RETRY_MAX_NUM_ATTEMPTS
from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.prompt_converter import TranslationConverter


def test_prompt_translation_init_templates_not_null():
    prompt_target = MockPromptTarget()
    translation_converter = TranslationConverter(converter_target=prompt_target, language="en")
    assert translation_converter.system_prompt


@pytest.mark.parametrize("languages", [None, ""])
def test_translator_converter_languages_validation_throws(languages):
    prompt_target = MockPromptTarget()
    with pytest.raises(ValueError):
        TranslationConverter(converter_target=prompt_target, language=languages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "converted_value",
    [
        "Invalid Json",
        "{'str' : 'json not formatted correctly'}",
    ],
)
async def test_translation_converter_send_prompt_async_bad_json_exception_retries(converted_value):
    prompt_target = MockPromptTarget()

    prompt_variation = TranslationConverter(converter_target=prompt_target, language="en")

    with patch("tests.mocks.MockPromptTarget.send_prompt_async", new_callable=AsyncMock) as mock_create:

        prompt_req_resp = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    conversation_id="12345679",
                    original_value="test input",
                    converted_value="this is not a json",
                    original_value_data_type="text",
                    converted_value_data_type="text",
                    prompt_target_identifier={"target": "target-identifier"},
                    orchestrator_identifier={"test": "test"},
                    labels={"test": "test"},
                )
            ]
        )
        mock_create.return_value = prompt_req_resp

        with pytest.raises(InvalidJsonException):
            await prompt_variation.convert_async(prompt="testing", input_type="text")
            assert mock_create.call_count == RETRY_MAX_NUM_ATTEMPTS


@pytest.mark.asyncio
async def test_translation_converter_send_prompt_async_json_bad_format_retries():
    prompt_target = MockPromptTarget()

    prompt_variation = TranslationConverter(converter_target=prompt_target, language="en")

    with patch("tests.mocks.MockPromptTarget.send_prompt_async", new_callable=AsyncMock) as mock_create:

        prompt_req_resp = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    conversation_id="12345679",
                    original_value="test input",
                    converted_value="this is not a json",
                    original_value_data_type="text",
                    converted_value_data_type="text",
                    prompt_target_identifier={"target": "target-identifier"},
                    orchestrator_identifier={"test": "test"},
                    labels={"test": "test"},
                )
            ]
        )
        mock_create.return_value = prompt_req_resp

        with pytest.raises(InvalidJsonException):
            await prompt_variation.convert_async(prompt="testing", input_type="text")
            assert mock_create.call_count == RETRY_MAX_NUM_ATTEMPTS
