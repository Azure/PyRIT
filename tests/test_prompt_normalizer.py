# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
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

    def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:  # type: ignore
        return ConverterResult(output_text=prompt, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"


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

    filename = ""

    with tempfile.NamedTemporaryFile(delete=False) as f:
        filename = f.name
        f.write(b"Hello")

        mock_image_converter.convert_async.return_value = ConverterResult(
            output_type="image_path",
            output_text=filename,
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
        assert sent_request.converted_value == filename
        assert sent_request.converted_value_data_type == "image_path"
    os.remove(filename)


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


@pytest.mark.asyncio
async def test_build_prompt_request_response():

    labels = {"label1": "value1", "label2": "value2"}
    orchestrator_identifier = {"orchestrator_id": "123"}

    prompt_target = MockPromptTarget()
    prompt_converters = [Base64Converter()]
    prompt_text = "Hello"
    normalizer_req_piece_1 = NormalizerRequestPiece(
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        prompt_data_type="text",
    )
    normalizer_req_piece_2 = NormalizerRequestPiece(
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        prompt_data_type="text",
    )
    normalizer = PromptNormalizer(memory=MagicMock())

    response = await normalizer._build_prompt_request_response(
        request=NormalizerRequest([normalizer_req_piece_1, normalizer_req_piece_2]),
        target=prompt_target,
        labels=labels,
        orchestrator_identifier=orchestrator_identifier,
    )

    # Check all prompt pieces in the response have the same conversation ID
    assert len(set(prompt_piece.conversation_id for prompt_piece in response.request_pieces)) == 1
