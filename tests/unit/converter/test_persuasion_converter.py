# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import AsyncMock, patch

import pytest
from unit.mocks import MockPromptTarget

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_converter import PersuasionConverter


def test_prompt_persuasion_init_authority_endorsement_template_not_null(duckdb_instance):
    prompt_target = MockPromptTarget()
    prompt_persuasion = PersuasionConverter(
        converter_target=prompt_target, persuasion_technique="authority_endorsement"
    )
    assert prompt_persuasion.system_prompt


def test_prompt_persuasion_init_evidence_based_template_not_null(duckdb_instance):
    prompt_target = MockPromptTarget()
    prompt_persuasion = PersuasionConverter(converter_target=prompt_target, persuasion_technique="evidence_based")
    assert prompt_persuasion.system_prompt


def test_prompt_persuasion_init_expert_endorsement_template_not_null(duckdb_instance):
    prompt_target = MockPromptTarget()
    prompt_persuasion = PersuasionConverter(converter_target=prompt_target, persuasion_technique="expert_endorsement")
    assert prompt_persuasion.system_prompt


def test_prompt_persuasion_init_logical_appeal_template_not_null(duckdb_instance):
    prompt_target = MockPromptTarget()
    prompt_persuasion = PersuasionConverter(converter_target=prompt_target, persuasion_technique="logical_appeal")
    assert prompt_persuasion.system_prompt


def test_prompt_persuasion_init_misrepresentation_template_not_null(duckdb_instance):
    prompt_target = MockPromptTarget()
    prompt_persuasion = PersuasionConverter(converter_target=prompt_target, persuasion_technique="misrepresentation")
    assert prompt_persuasion.system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "converted_value",
    [
        "Invalid Json",
        "{'str' : 'json not formatted correctly'}",
    ],
)
async def test_persuasion_converter_send_prompt_async_bad_json_exception_retries(converted_value, duckdb_instance):

    prompt_target = MockPromptTarget()

    prompt_persuasion = PersuasionConverter(
        converter_target=prompt_target, persuasion_technique="authority_endorsement"
    )

    with patch("unit.mocks.MockPromptTarget.send_prompt_async", new_callable=AsyncMock) as mock_create:

        prompt_req_resp = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    conversation_id="12345679",
                    original_value="test input",
                    converted_value=converted_value,
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
            await prompt_persuasion.convert_async(prompt="testing", input_type="text")
            assert mock_create.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


def test_persuasion_converter_input_supported():
    prompt_target = MockPromptTarget()
    prompt_persuasion = PersuasionConverter(
        converter_target=prompt_target, persuasion_technique="authority_endorsement"
    )
    assert prompt_persuasion.input_supported("text") is True
    assert prompt_persuasion.input_supported("image_path") is False
