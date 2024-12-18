# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.models import PromptDataType
from pyrit.prompt_normalizer import NormalizerRequestPiece
from pyrit.prompt_converter import PromptConverter, ConverterResult


class MockPromptConverter(PromptConverter):

    def __init__(self) -> None:
        pass

    async def convert_async(self, *, prompt, input_type: PromptDataType = "text") -> ConverterResult:  # type: ignore
        return ConverterResult(output_text=prompt, output_type="text")  # type: ignore

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"


def test_prompt_request_piece_init_valid_arguments():
    prompt_converters = [MockPromptConverter()]
    prompt_text = "Hello"
    metadata = "meta"

    prompt = NormalizerRequestPiece(
        request_converters=prompt_converters,
        prompt_value=prompt_text,
        prompt_data_type="text",
        metadata=metadata,
    )

    assert prompt.request_converters == prompt_converters
    assert prompt.prompt_value == prompt_text
    assert prompt.prompt_data_type == "text"
    assert prompt.metadata == metadata


def test_prompt_init_no_metadata():
    prompt_converters = [MockPromptConverter()]
    prompt_text = "Hello"

    prompt = NormalizerRequestPiece(
        request_converters=prompt_converters,
        prompt_value=prompt_text,
        prompt_data_type="text",
    )

    assert prompt.request_converters == prompt_converters
    assert prompt.prompt_value == prompt_text
    assert prompt.prompt_data_type == "text"
    assert not prompt.metadata


def test_prompt_request_piece_init_invalid_converter():
    prompt_text = "Hello"
    metadata = "meta"

    with pytest.raises(ValueError):
        NormalizerRequestPiece(
            request_converters=["InvalidPromptConverter"],
            prompt_value=prompt_text,
            prompt_data_type="text",
            metadata=metadata,
        )


def test_prompt_init_invalid_prompt_text():
    metadata = "meta"

    with pytest.raises(ValueError):
        NormalizerRequestPiece(
            request_converters=[],
            prompt_value=None,
            prompt_data_type="text",
            metadata=metadata,
        )
