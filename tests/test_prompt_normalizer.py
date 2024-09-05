# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from pyrit.models import PromptDataType
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse
from pyrit.prompt_converter import Base64Converter, StringJoinConverter
from pyrit.prompt_normalizer import NormalizerRequestPiece, NormalizerRequest, PromptNormalizer
from pyrit.prompt_converter import PromptConverter, ConverterResult

from pyrit.prompt_normalizer.prompt_response_converter_configuration import PromptResponseConverterConfiguration
from pyrit.prompt_target.prompt_target import PromptTarget
from tests.mocks import MockPromptTarget, get_image_request_piece


@pytest.fixture
def response() -> PromptRequestResponse:
    image_request_piece = get_image_request_piece()
    image_request_piece.role = "assistant"
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value="Hello",
            ),
            PromptRequestPiece(role="assistant", original_value="part 2"),
            image_request_piece,
        ]
    )


@pytest.fixture
def normalizer_piece() -> NormalizerRequestPiece:
    return NormalizerRequestPiece(
        request_converters=[],
        prompt_value="Hello",
        prompt_data_type="text",
    )


class MockPromptConverter(PromptConverter):

    def __init__(self) -> None:
        pass

    def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:  # type: ignore
        return ConverterResult(output_text=prompt, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"


@pytest.mark.asyncio
async def test_send_prompt_async_multiple_converters(normalizer_piece: NormalizerRequestPiece):
    prompt_target = MockPromptTarget()
    normalizer_piece.request_converters = [Base64Converter(), StringJoinConverter(join_value="_")]

    request = NormalizerRequest([normalizer_piece])

    normalizer = PromptNormalizer(memory=MagicMock())

    await normalizer.send_prompt_async(normalizer_request=request, target=prompt_target)

    assert prompt_target.prompt_sent == ["S_G_V_s_b_G_8_="]


@pytest.mark.asyncio
async def test_send_prompt_async_no_response_adds_memory(normalizer_piece: NormalizerRequestPiece):
    prompt_target = AsyncMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=None)

    memory = MagicMock()

    normalizer = PromptNormalizer(memory=memory)

    await normalizer.send_prompt_async(normalizer_request=NormalizerRequest([normalizer_piece]), target=prompt_target)

    assert memory.add_request_response_to_memory.call_count == 1


@pytest.mark.asyncio
async def test_send_prompt_async_request_response_added_to_memory(normalizer_piece: NormalizerRequestPiece):
    prompt_target = AsyncMock()

    response = PromptRequestPiece(role="assistant", original_value="test_response").to_prompt_request_response()

    prompt_target.send_prompt_async = AsyncMock(return_value=response)
    memory = MagicMock()
    normalizer = PromptNormalizer(memory=memory)

    await normalizer.send_prompt_async(normalizer_request=NormalizerRequest([normalizer_piece]), target=prompt_target)

    assert memory.add_request_response_to_memory.call_count == 2

    # Validate that first request is added to memory, then response is added to memory
    assert (
        normalizer_piece.prompt_value
        == memory.add_request_response_to_memory.call_args_list[0][1]["request"].request_pieces[0].original_value
    )
    assert (
        "test_response"
        == memory.add_request_response_to_memory.call_args_list[1][1]["request"].request_pieces[0].original_value
    )

    assert memory.add_request_response_to_memory.call_args_list[1].called_after(prompt_target.send_prompt_async)


@pytest.mark.asyncio
async def test_send_prompt_async_exception(normalizer_piece: NormalizerRequestPiece):
    prompt_target = AsyncMock()
    prompt_target.send_prompt_async = AsyncMock(side_effect=ValueError("test_exception"))

    memory = MagicMock()

    normalizer = PromptNormalizer(memory=memory)
    await normalizer._build_prompt_request_response(request=NormalizerRequest([normalizer_piece]), target=prompt_target)

    with patch("pyrit.models.construct_response_from_request") as mock_construct:
        mock_construct.return_value = "test"

        try:
            await normalizer.send_prompt_async(
                normalizer_request=NormalizerRequest([normalizer_piece]), target=prompt_target
            )
        except ValueError:
            assert memory.add_request_response_to_memory.call_count == 2

            # Validate that first request is added to memory, then exception is added to memory
            assert (
                normalizer_piece.prompt_value
                == memory.add_request_response_to_memory.call_args_list[0][1]["request"]
                .request_pieces[0]
                .original_value
            )
            assert (
                "test_exception"
                == memory.add_request_response_to_memory.call_args_list[1][1]["request"]
                .request_pieces[0]
                .original_value
            )


@pytest.mark.asyncio
async def test_send_prompt_async_adds_memory_twice(
    normalizer_piece: NormalizerRequestPiece, response: PromptRequestResponse
):

    prompt_target = MagicMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=response)

    normalizer_piece.request_converters = []

    request = NormalizerRequest(request_pieces=[normalizer_piece])
    memory = MagicMock()
    normalizer = PromptNormalizer(memory=memory)

    response = await normalizer.send_prompt_async(normalizer_request=request, target=prompt_target)
    assert memory.add_request_response_to_memory.call_count == 2


@pytest.mark.asyncio
async def test_send_prompt_async_no_converters_response(
    normalizer_piece: NormalizerRequestPiece, response: PromptRequestResponse
):

    prompt_target = MagicMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=response)

    normalizer_piece.request_converters = []
    request = NormalizerRequest([normalizer_piece])

    memory = MagicMock()

    normalizer = PromptNormalizer(memory=memory)

    response = await normalizer.send_prompt_async(normalizer_request=request, target=prompt_target)
    assert response.request_pieces[0].converted_value == "Hello", "There were no response converters"


@pytest.mark.asyncio
async def test_send_prompt_async_converters_response(
    normalizer_piece: NormalizerRequestPiece, response: PromptRequestResponse
):

    prompt_target = MagicMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=response)

    normalizer_piece.request_converters = []

    response_converter = PromptResponseConverterConfiguration(converters=[Base64Converter()], indexes_to_apply=[0])

    request = NormalizerRequest(request_pieces=[normalizer_piece], response_converters=[response_converter])

    memory = MagicMock()

    normalizer = PromptNormalizer(memory=memory)

    response = await normalizer.send_prompt_async(normalizer_request=request, target=prompt_target)
    assert response.request_pieces[0].converted_value == "SGVsbG8="


@pytest.mark.asyncio
async def test_send_prompt_async_image_converter():
    prompt_target = MagicMock(PromptTarget)

    mock_image_converter = MagicMock(PromptConverter)

    filename = ""

    with tempfile.NamedTemporaryFile(delete=False) as f:
        filename = f.name
        f.write(b"Hello")

        mock_image_converter.convert_tokens_async.return_value = ConverterResult(
            output_type="image_path",
            output_text=filename,
        )

        prompt_converters = [mock_image_converter]
        prompt_text = "Hello"

        prompt = NormalizerRequestPiece(
            request_converters=prompt_converters,
            prompt_value=prompt_text,
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
@pytest.mark.parametrize("max_requests_per_minute", [None, 10])
@pytest.mark.parametrize("batch_size", [1, 10])
async def test_prompt_normalizer_send_prompt_batch_async_throws(
    normalizer_piece: NormalizerRequestPiece,
    max_requests_per_minute: int,
    batch_size: int,
):
    prompt_target = MockPromptTarget(rpm=max_requests_per_minute)

    normalizer_piece.request_converters = [Base64Converter(), StringJoinConverter(join_value="_")]
    normalizer = PromptNormalizer(memory=MagicMock())

    if max_requests_per_minute and batch_size != 1:
        with pytest.raises(ValueError):
            results = await normalizer.send_prompt_batch_to_target_async(
                requests=[NormalizerRequest([normalizer_piece])],
                target=prompt_target,
                batch_size=batch_size,
            )
    else:
        results = await normalizer.send_prompt_batch_to_target_async(
            requests=[NormalizerRequest([normalizer_piece])],
            target=prompt_target,
            batch_size=batch_size,
        )

        assert "S_G_V_s_b_G_8_=" in prompt_target.prompt_sent
        assert len(results) == 1


@pytest.mark.asyncio
async def test_build_prompt_request_response():

    labels = {"label1": "value1", "label2": "value2"}
    orchestrator_identifier = {"orchestrator_id": "123"}

    prompt_target = MockPromptTarget()
    prompt_converters = [Base64Converter()]
    prompt_text = "Hello"
    normalizer_req_piece_1 = NormalizerRequestPiece(
        request_converters=prompt_converters,
        prompt_value=prompt_text,
        prompt_data_type="text",
    )
    normalizer_req_piece_2 = NormalizerRequestPiece(
        request_converters=prompt_converters,
        prompt_value=prompt_text,
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


@pytest.mark.asyncio
async def test_convert_response_values_index(response: PromptRequestResponse):
    response_converter = PromptResponseConverterConfiguration(converters=[Base64Converter()], indexes_to_apply=[0])

    normalizer = PromptNormalizer(memory=MagicMock())

    await normalizer.convert_response_values(
        response_converter_configurations=[response_converter], prompt_response=response
    )
    assert response.request_pieces[0].converted_value == "SGVsbG8=", "Converter should be applied here"
    assert (
        response.request_pieces[1].converted_value == "part 2"
    ), "Converter should not be applied since we specified only 0"


@pytest.mark.asyncio
async def test_convert_response_values_type(response: PromptRequestResponse):
    response_converter = PromptResponseConverterConfiguration(
        converters=[Base64Converter()], prompt_data_types_to_apply=["text"]
    )

    normalizer = PromptNormalizer(memory=MagicMock())

    await normalizer.convert_response_values(
        response_converter_configurations=[response_converter], prompt_response=response
    )
    assert response.request_pieces[0].converted_value == "SGVsbG8="
    assert response.request_pieces[1].converted_value == "cGFydCAy"
