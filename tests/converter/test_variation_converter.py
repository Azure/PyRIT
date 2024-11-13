# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from tests.mocks import MockPromptTarget

import pytest
import os
from unittest.mock import AsyncMock, patch

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse
from pyrit.prompt_converter import VariationConverter
from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


def test_prompt_variation_init_templates_not_null(memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
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
async def test_variation_converter_send_prompt_async_bad_json_exception_retries(
    converted_value, memory_interface: MemoryInterface
):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        prompt_target = MockPromptTarget()

        prompt_variation = VariationConverter(converter_target=prompt_target)

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
                await prompt_variation.convert_async(prompt="testing", input_type="text")

            assert mock_create.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


def test_variation_converter_input_supported(memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        prompt_target = MockPromptTarget()
        converter = VariationConverter(converter_target=prompt_target)
        assert converter.input_supported("audio_path") is False
        assert converter.input_supported("text") is True
