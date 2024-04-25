# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock
import pytest

from pyrit.models import PromptDataType
from pyrit.prompt_converter import Base64Converter, StringJoinConverter
from pyrit.prompt_normalizer import NormalizerRequestPiece, NormalizerRequest, PromptNormalizer
from pyrit.prompt_converter import PromptConverter, ConverterResult

from pyrit.prompt_target.prompt_target import PromptTarget
from tests.mocks import MockPromptTarget


class MockPromptConverter(PromptConverter):

    def __init__(self) -> None:
        pass

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        return ConverterResult(output_text=prompt, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"


def test_send_prompt_multiple_converters():
    prompt_target = MockPromptTarget()
    prompt_converters = [Base64Converter(), StringJoinConverter(join_value="_")]
    prompt_text = "Hello"

    prompt = NormalizerRequestPiece(
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        prompt_data_type="text",
        metadata="metadata",
    )

    normalizer = PromptNormalizer(memory=MagicMock())

    normalizer.send_prompt(normalizer_request=NormalizerRequest([prompt]), target=prompt_target)

    assert prompt_target.prompt_sent == ["S_G_V_s_b_G_8_="]


@pytest.mark.asyncio
async def test_send_prompt_async_multiple_converters():
    prompt_target = MockPromptTarget()
    prompt_converters = [Base64Converter(), StringJoinConverter(join_value="_")]
    prompt_text = "Hello"

    prompt = NormalizerRequestPiece(
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        prompt_data_type="text",
    )

    normalizer = PromptNormalizer(memory=MagicMock())

    await normalizer.send_prompt_async(normalizer_request=NormalizerRequest([prompt]), target=prompt_target)

    assert prompt_target.prompt_sent == ["S_G_V_s_b_G_8_="]


@pytest.mark.asyncio
async def test_send_prompt_async_image_converter():
    prompt_target = MagicMock(PromptTarget)

    mock_image_converter = MagicMock(PromptConverter)

    mock_image_converter.convert.return_value = ConverterResult(
        output_type="path_to_image",
        output_text="image_path",
    )

    prompt_converters = [mock_image_converter]
    prompt_text = "Hello"

    prompt = NormalizerRequestPiece(
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        prompt_data_type="text",
    )

    normalizer = PromptNormalizer(memory=MagicMock())

    await normalizer.send_prompt_async(normalizer_request=NormalizerRequest([prompt]), target=prompt_target)

    # verify the prompt target received the correct arguments from the normalizer
    sent_request = prompt_target.send_prompt_async.call_args.kwargs["prompt_request"].request_pieces[0]
    assert sent_request.converted_prompt_text == "image_path"
    assert sent_request.converted_prompt_data_type == "path_to_image"


@pytest.mark.asyncio
async def test_prompt_normalizer_send_prompt_batch_async():
    prompt_target = MockPromptTarget()
    prompt_converters = [Base64Converter(), StringJoinConverter(join_value="_")]
    prompt_text = "Hello"

    prompt = NormalizerRequestPiece(
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        prompt_data_type="text",
    )

    normalizer = PromptNormalizer(memory=MagicMock())

    await normalizer.send_prompt_batch_to_target_async(requests=[NormalizerRequest([prompt])], target=prompt_target)

    assert prompt_target.prompt_sent == ["S_G_V_s_b_G_8_="]
