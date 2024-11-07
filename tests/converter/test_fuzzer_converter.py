# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Generator
import pytest
from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from tests.mocks import MockPromptTarget
from unittest.mock import AsyncMock, patch

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse
from pyrit.prompt_converter import (
    FuzzerExpandConverter,
    FuzzerShortenConverter,
    FuzzerRephraseConverter,
    FuzzerCrossOverConverter,
    FuzzerSimilarConverter,
)
from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.parametrize(
    "converter_class",
    [
        FuzzerExpandConverter,
        FuzzerShortenConverter,
        FuzzerRephraseConverter,
        FuzzerCrossOverConverter,
        FuzzerSimilarConverter,
    ],
)
def test_converter_init_templates_not_null(converter_class, memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
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
    "converter_class",
    [
        FuzzerExpandConverter,
        FuzzerShortenConverter,
        FuzzerRephraseConverter,
        FuzzerCrossOverConverter,
        FuzzerSimilarConverter,
    ],
)
@pytest.mark.parametrize(
    "update",
    [True, False],
)
async def test_converter_send_prompt_async_bad_json_exception_retries(
    converted_value, converter_class, update, memory_interface
):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        prompt_target = MockPromptTarget()

        if converter_class != FuzzerCrossOverConverter:
            converter = converter_class(converter_target=prompt_target)
        else:
            converter = converter_class(converter_target=prompt_target, prompt_templates=["testing 1"])

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

            if update:
                converter.update(prompt_templates=["testing 2"])

            if converter_class == FuzzerCrossOverConverter:
                if update:
                    assert converter.prompt_templates == ["testing 2"]
                else:
                    assert converter.prompt_templates == ["testing 1"]

            with pytest.raises(InvalidJsonException):
                await converter.convert_async(prompt="testing", input_type="text")
                assert int(os.getenv("RETRY_MAX_NUM_ATTEMPTS")) == 2
                assert mock_create.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.parametrize(
    "converter_class",
    [
        FuzzerExpandConverter,
        FuzzerShortenConverter,
        FuzzerRephraseConverter,
        FuzzerCrossOverConverter,
        FuzzerSimilarConverter,
    ],
)
def test_fuzzer_converter_input_supported(converter_class, memory_interface) -> None:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        prompt_target = MockPromptTarget()
        converter = converter_class(converter_target=prompt_target)
        assert converter.input_supported("text") is True
        assert converter.input_supported("image_path") is False
