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
from pyrit.prompt_converter import TranslationConverter
from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


def test_prompt_translation_init_templates_not_null(memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        prompt_target = MockPromptTarget()
        translation_converter = TranslationConverter(converter_target=prompt_target, language="en")
        assert translation_converter.system_prompt


@pytest.mark.parametrize("languages", [None, ""])
def test_translator_converter_languages_validation_throws(languages, memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
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
async def test_translation_converter_send_prompt_async_bad_json_exception_retries(
    converted_value, memory_interface: MemoryInterface
):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):

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
                assert mock_create.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_translation_converter_send_prompt_async_json_bad_format_retries(memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):

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
                assert mock_create.call_count == os.getenv("RETRY_MAX_NUM_ATTEMPTS")


@pytest.mark.asyncio
async def test_translation_converter_convert_async_retrieve_key_capitalization_mismatch(
    memory_interface: MemoryInterface,
):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        prompt_target = MockPromptTarget()

        translation_converter = TranslationConverter(converter_target=prompt_target, language="spanish")
        with patch.object(
            translation_converter, "send_translation_prompt_async", new=AsyncMock(return_value={"Spanish": "hola"})
        ):

            raised = False
            try:
                await translation_converter.convert_async(prompt="hello")
            except KeyError:
                raised = True  # There should be no KeyError

            assert raised is False


def test_translation_converter_input_supported(memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        prompt_target = MockPromptTarget()
        translation_converter = TranslationConverter(converter_target=prompt_target, language="spanish")
        assert translation_converter.input_supported("text") is True
        assert translation_converter.input_supported("image_path") is False
