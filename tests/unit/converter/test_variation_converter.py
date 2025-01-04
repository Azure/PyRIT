# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import AsyncMock, patch

import pytest
from unit.mocks import MockPromptTarget

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_converter import VariationConverter


def test_prompt_variation_init_templates_not_null(duckdb_instance):
    prompt_target = MockPromptTarget()
    prompt_variation = VariationConverter(converter_target=prompt_target)
    assert prompt_variation.system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "converted_value",
    [
        "Invalid Json",
        "{'str' : 'json not formatted correctly'}",
    ],
)
async def test_variation_converter_send_prompt_async_bad_json_exception_retries(converted_value, duckdb_instance):
    prompt_target = MockPromptTarget()

    prompt_variation = VariationConverter(converter_target=prompt_target)

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
            await prompt_variation.convert_async(prompt="testing", input_type="text")

        assert mock_create.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


def test_variation_converter_input_supported(duckdb_instance):
    prompt_target = MockPromptTarget()
    converter = VariationConverter(converter_target=prompt_target)
    assert converter.input_supported("audio_path") is False
    assert converter.input_supported("text") is True
