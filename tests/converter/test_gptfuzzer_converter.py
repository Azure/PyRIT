# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest
from tests.mocks import MockPromptTarget
from unittest.mock import AsyncMock, patch

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse
from pyrit.prompt_converter import (
    FuzzerCrossOverConverter,
    FuzzerShortenConverter,
    FuzzerRephraseConverter,
    FuzzerCrossOverConverter,
    FuzzerRephraseConverter,
)


@pytest.mark.parametrize(
    "converter_class", [FuzzerCrossOverConverter, FuzzerShortenConverter, FuzzerRephraseConverter, FuzzerCrossOverConverter, FuzzerRephraseConverter]
)
def test_converter_init_templates_not_null(converter_class):
    prompt_target = MockPromptTarget()
    converter = converter_class(converter_target=prompt_target)
    assert converter.system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "converted_value",
    [
        "Invalid Json",
        "{'str' : 'json not formatted correctly'}",
    ],
)
@pytest.mark.parametrize(
    "converter_class", [FuzzerCrossOverConverter, FuzzerShortenConverter, FuzzerRephraseConverter, FuzzerCrossOverConverter, FuzzerRephraseConverter]
)
async def test_converter_send_prompt_async_bad_json_exception_retries(converted_value, converter_class):
    prompt_target = MockPromptTarget()

    if converter_class != FuzzerCrossOverConverter:
        converter = converter_class(converter_target=prompt_target)
    else:
        converter = converter_class(converter_target=prompt_target, prompts=["testing"])

    with patch("tests.mocks.MockPromptTarget.send_prompt_async", new_callable=AsyncMock) as mock_create:

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
            await converter.convert_async(prompt="testing", input_type="text")
            assert int(os.getenv("RETRY_MAX_NUM_ATTEMPTS")) == 2
            assert mock_create.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))
