# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import MockPromptTarget, get_image_request_piece

from pyrit.exceptions import EmptyResponseException
from pyrit.memory import CentralMemory
from pyrit.models import PromptDataType, PromptRequestPiece, PromptRequestResponse
from pyrit.models.filter_criteria import PromptFilterCriteria
from pyrit.models.seed_prompt import SeedPrompt, SeedPromptGroup
from pyrit.prompt_converter import (
    Base64Converter,
    ConverterResult,
    PromptConverter,
    StringJoinConverter,
)
from pyrit.prompt_normalizer import NormalizerRequest, PromptNormalizer
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptTarget


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
def seed_prompt_group() -> SeedPromptGroup:
    return SeedPromptGroup(
        prompts=[
            SeedPrompt(
                value="Hello",
                data_type="text",
            )
        ]
    )


@pytest.fixture
def mock_memory_instance():
    """Fixture to mock CentralMemory.get_memory_instance"""
    memory = MagicMock()
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        yield memory


class MockPromptConverter(PromptConverter):

    def __init__(self) -> None:
        pass

    def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:  # type: ignore
        return ConverterResult(output_text=prompt, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"


def assert_prompt_piece_hashes_set(request: PromptRequestResponse):
    assert request
    assert request.request_pieces
    for piece in request.request_pieces:
        assert piece.original_value_sha256
        assert piece.converted_value_sha256


@pytest.mark.asyncio
async def test_send_prompt_async_multiple_converters(mock_memory_instance, seed_prompt_group):
    prompt_target = MockPromptTarget()
    request_converters = [
        PromptConverterConfiguration(converters=[Base64Converter(), StringJoinConverter(join_value="_")])
    ]

    normalizer = PromptNormalizer()

    await normalizer.send_prompt_async(
        seed_prompt_group=seed_prompt_group, request_converter_configurations=request_converters, target=prompt_target
    )

    assert prompt_target.prompt_sent == ["S_G_V_s_b_G_8_="]


@pytest.mark.asyncio
async def test_send_prompt_async_no_response_adds_memory(mock_memory_instance, seed_prompt_group):
    prompt_target = AsyncMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=None)

    normalizer = PromptNormalizer()

    await normalizer.send_prompt_async(seed_prompt_group=seed_prompt_group, target=prompt_target)
    assert mock_memory_instance.add_request_response_to_memory.call_count == 1

    request = mock_memory_instance.add_request_response_to_memory.call_args[1]["request"]
    assert_prompt_piece_hashes_set(request)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_exception_handled(mock_memory_instance, seed_prompt_group):
    prompt_target = AsyncMock()
    prompt_target.send_prompt_async = AsyncMock(side_effect=EmptyResponseException(message="Empty response"))

    normalizer = PromptNormalizer()

    response = await normalizer.send_prompt_async(seed_prompt_group=seed_prompt_group, target=prompt_target)

    assert mock_memory_instance.add_request_response_to_memory.call_count == 2

    assert response.request_pieces[0].response_error == "empty"
    assert response.request_pieces[0].original_value == ""
    assert response.request_pieces[0].original_value_data_type == "text"

    assert_prompt_piece_hashes_set(response)


@pytest.mark.asyncio
async def test_send_prompt_async_request_response_added_to_memory(mock_memory_instance, seed_prompt_group):
    prompt_target = AsyncMock()

    response = PromptRequestPiece(role="assistant", original_value="test_response").to_prompt_request_response()

    prompt_target.send_prompt_async = AsyncMock(return_value=response)

    normalizer = PromptNormalizer()

    await normalizer.send_prompt_async(seed_prompt_group=seed_prompt_group, target=prompt_target)

    assert mock_memory_instance.add_request_response_to_memory.call_count == 2

    seed_prompt_value = seed_prompt_group.prompts[0].value
    # Validate that first request is added to memory, then response is added to memory
    assert (
        seed_prompt_value
        == mock_memory_instance.add_request_response_to_memory.call_args_list[0][1]["request"]
        .request_pieces[0]
        .original_value
    )
    assert (
        "test_response"
        == mock_memory_instance.add_request_response_to_memory.call_args_list[1][1]["request"]
        .request_pieces[0]
        .original_value
    )

    assert mock_memory_instance.add_request_response_to_memory.call_args_list[1].called_after(
        prompt_target.send_prompt_async
    )


@pytest.mark.asyncio
async def test_send_prompt_async_exception(mock_memory_instance, seed_prompt_group):
    prompt_target = AsyncMock()

    seed_prompt_value = seed_prompt_group.prompts[0].value

    normalizer = PromptNormalizer()

    with patch("pyrit.models.construct_response_from_request") as mock_construct:
        mock_construct.return_value = "test"

        try:
            await normalizer.send_prompt_async(seed_prompt_group=seed_prompt_group, target=prompt_target)
        except ValueError:
            assert mock_memory_instance.add_request_response_to_memory.call_count == 2

            # Validate that first request is added to memory, then exception is added to memory
            assert (
                seed_prompt_value
                == mock_memory_instance.add_request_response_to_memory.call_args_list[0][1]["request"]
                .request_pieces[0]
                .original_value
            )
            assert (
                "test_exception"
                == mock_memory_instance.add_request_response_to_memory.call_args_list[1][1]["request"]
                .request_pieces[0]
                .original_value
            )


@pytest.mark.asyncio
async def test_send_prompt_async_empty_exception(mock_memory_instance, seed_prompt_group):
    prompt_target = AsyncMock()
    prompt_target.send_prompt_async = AsyncMock(side_effect=Exception(""))

    normalizer = PromptNormalizer()

    with pytest.raises(Exception, match="Error sending prompt with conversation ID"):
        await normalizer.send_prompt_async(seed_prompt_group=seed_prompt_group, target=prompt_target)


@pytest.mark.asyncio
async def test_send_prompt_async_adds_memory_twice(
    mock_memory_instance, seed_prompt_group, response: PromptRequestResponse
):
    prompt_target = MagicMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=response)

    normalizer = PromptNormalizer()

    response = await normalizer.send_prompt_async(seed_prompt_group=seed_prompt_group, target=prompt_target)
    assert mock_memory_instance.add_request_response_to_memory.call_count == 2


@pytest.mark.asyncio
async def test_send_prompt_async_no_converters_response(
    mock_memory_instance, seed_prompt_group, response: PromptRequestResponse
):

    prompt_target = MagicMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=response)

    normalizer = PromptNormalizer()

    # Send prompt async and check the response
    response = await normalizer.send_prompt_async(seed_prompt_group=seed_prompt_group, target=prompt_target)
    assert response.get_value() == "Hello", "There were no response converters"


@pytest.mark.asyncio
async def test_send_prompt_async_converters_response(
    mock_memory_instance, seed_prompt_group, response: PromptRequestResponse
):

    prompt_target = MagicMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=response)

    response_converter = PromptConverterConfiguration(converters=[Base64Converter()], indexes_to_apply=[0])

    normalizer = PromptNormalizer()

    response = await normalizer.send_prompt_async(
        seed_prompt_group=seed_prompt_group,
        response_converter_configurations=[response_converter],
        target=prompt_target,
    )

    assert response.get_value() == "SGVsbG8="


@pytest.mark.asyncio
async def test_send_prompt_async_image_converter(mock_memory_instance):
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

        prompt_converters = PromptConverterConfiguration(converters=[mock_image_converter])

        prompt_text = "Hello"

        seed_prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value=prompt_text, data_type="text")])

        normalizer = PromptNormalizer()
        # Mock the async read_file method
        normalizer._memory.results_storage_io.read_file = AsyncMock(return_value=b"mocked data")

        response = await normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt_group,
            target=prompt_target,
            request_converter_configurations=[prompt_converters],
        )

        # verify the prompt target received the correct arguments from the normalizer
        sent_request = prompt_target.send_prompt_async.call_args.kwargs["prompt_request"].request_pieces[0]
        assert sent_request.converted_value == filename
        assert sent_request.converted_value_data_type == "image_path"

        assert_prompt_piece_hashes_set(response)
    os.remove(filename)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_requests_per_minute", [None, 10])
@pytest.mark.parametrize("batch_size", [1, 10])
async def test_prompt_normalizer_send_prompt_batch_async_throws(
    mock_memory_instance, seed_prompt_group, max_requests_per_minute, batch_size
):
    prompt_target = MockPromptTarget(rpm=max_requests_per_minute)

    request_converters = PromptConverterConfiguration(
        converters=[Base64Converter(), StringJoinConverter(join_value="_")]
    )

    normalizer_request = NormalizerRequest(
        seed_prompt_group=seed_prompt_group,
        request_converter_configurations=[request_converters],
    )

    normalizer = PromptNormalizer()

    if max_requests_per_minute and batch_size != 1:
        with pytest.raises(ValueError):
            results = await normalizer.send_prompt_batch_to_target_async(
                requests=[normalizer_request],
                target=prompt_target,
                batch_size=batch_size,
            )
    else:
        results = await normalizer.send_prompt_batch_to_target_async(
            requests=[normalizer_request],
            target=prompt_target,
            batch_size=batch_size,
        )

        assert "S_G_V_s_b_G_8_=" in prompt_target.prompt_sent
        assert len(results) == 1


@pytest.mark.asyncio
async def test_build_prompt_request_response(mock_memory_instance, seed_prompt_group):

    labels = {"label1": "value1", "label2": "value2"}
    orchestrator_identifier = {"orchestrator_id": "123"}

    conversation_id = uuid.uuid4()

    prompt_target = MockPromptTarget()
    request_converters = [
        PromptConverterConfiguration(converters=[Base64Converter(), StringJoinConverter(join_value="_")])
    ]

    normalizer = PromptNormalizer()

    response = await normalizer._build_prompt_request_response(
        seed_prompt_group=seed_prompt_group,
        conversation_id=conversation_id,
        request_converter_configurations=request_converters,
        target=prompt_target,
        sequence=2,
        labels=labels,
        orchestrator_identifier=orchestrator_identifier,
    )

    # Check all prompt pieces in the response have the same conversation ID
    assert len(set(prompt_piece.conversation_id for prompt_piece in response.request_pieces)) == 1

    # Check sequence is set correctly
    assert len(set(prompt_piece.sequence for prompt_piece in response.request_pieces)) == 1


@pytest.mark.asyncio
async def test_convert_response_values_index(mock_memory_instance, response: PromptRequestResponse):
    response_converter = PromptConverterConfiguration(converters=[Base64Converter()], indexes_to_apply=[0])

    normalizer = PromptNormalizer()

    await normalizer.convert_values(converter_configurations=[response_converter], request_response=response)
    assert response.get_value() == "SGVsbG8=", "Converter should be applied here"
    assert response.get_value(1) == "part 2", "Converter should not be applied since we specified only 0"


@pytest.mark.asyncio
async def test_convert_response_values_type(mock_memory_instance, response: PromptRequestResponse):
    response_converter = PromptConverterConfiguration(
        converters=[Base64Converter()], prompt_data_types_to_apply=["text"]
    )

    normalizer = PromptNormalizer()

    await normalizer.convert_values(converter_configurations=[response_converter], request_response=response)
    assert response.get_value() == "SGVsbG8="
    assert response.get_value(1) == "cGFydCAy"


@pytest.mark.asyncio
async def test_should_skip_based_on_skip_criteria_no_skip_criteria(mock_memory_instance):
    normalizer = PromptNormalizer()  # By default, _skip_criteria is None

    # Make a request with at least one piece
    request = PromptRequestResponse(request_pieces=[PromptRequestPiece(role="user", original_value="hello")])

    result = normalizer._should_skip_based_on_skip_criteria(request)
    assert result is False, "_should_skip_based_on_skip_criteria should return False when skip_criteria is not set"


@pytest.mark.asyncio
async def test_should_skip_based_on_skip_criteria_no_matches(mock_memory_instance):
    normalizer = PromptNormalizer()

    skip_criteria = PromptFilterCriteria(
        orchestrator_id="test_orchestrator",
        conversation_id="test_conversation",
    )

    memory_piece = PromptRequestPiece(
        role="user",
        original_value="My user prompt",
    )
    memory_piece.original_value_sha256 = "some_random_hash"
    memory_piece.converted_value_sha256 = "some random hash"

    mock_memory_instance.get_prompt_request_pieces.return_value = [memory_piece]

    normalizer.set_skip_criteria(skip_criteria, skip_value_type="converted")

    # Construct a request piece that doesn't match the memory's hash
    request_piece = PromptRequestPiece(role="user", original_value="My user prompt")
    request_piece.original_value_sha256 = "completely_different_hash"
    request_piece.converted_value_sha256 = "completely_different_hash"

    request = PromptRequestResponse(request_pieces=[request_piece])

    result = normalizer._should_skip_based_on_skip_criteria(request)
    assert result is False, "Should return False if no prompt pieces in memory match"


@pytest.mark.asyncio
async def test_should_skip_based_on_skip_criteria_match_found(mock_memory_instance):
    """
    If skip criteria is set and the prompt pieces in memory DO match,
    _should_skip_based_on_skip_criteria should return True.
    """
    normalizer = PromptNormalizer()

    skip_criteria = PromptFilterCriteria(
        orchestrator_id="test_orchestrator",
        conversation_id="test_conversation",
    )

    # We'll say that memory returns one piece with the exact same converted_value_sha256
    # as our request piece
    matching_sha = "matching_converted_hash"

    piece = PromptRequestPiece(role="user", original_value="prompt")
    piece.converted_value_sha256 = matching_sha
    mock_memory_instance.get_prompt_request_pieces.return_value = [piece]

    # Our request piece also has that same matching sha
    request_piece = PromptRequestPiece(role="user", original_value="My user prompt")
    request_piece.converted_value_sha256 = matching_sha

    request = PromptRequestResponse(request_pieces=[request_piece])

    # Set skip criteria with 'converted' skip_value_type
    normalizer.set_skip_criteria(skip_criteria, skip_value_type="converted")

    result = normalizer._should_skip_based_on_skip_criteria(request)
    assert result is True, "Should return True if a matching converted_value_sha256 is found"


@pytest.mark.asyncio
async def test_should_skip_based_on_skip_criteria_original_value_match(mock_memory_instance):
    """
    Same test logic but with skip_value_type='original'.
    """
    matching_sha = "matching_original_hash"

    # Build a request piece with the same original_value_sha256
    request_piece = PromptRequestPiece(role="user", original_value="My user prompt")
    request_piece.original_value_sha256 = matching_sha

    request = PromptRequestResponse(request_pieces=[request_piece])

    # Memory returns a piece that has an original_value_sha256 matching our request piece
    piece = PromptRequestPiece(role="user", original_value="prompt")
    piece.original_value_sha256 = matching_sha
    mock_memory_instance.get_prompt_request_pieces.return_value = [piece]

    normalizer = PromptNormalizer()

    skip_criteria = PromptFilterCriteria(
        orchestrator_id="test_orchestrator",
        conversation_id="test_conversation",
    )

    # This time we use 'original' skip_value_type
    normalizer.set_skip_criteria(skip_criteria, skip_value_type="original")

    result = normalizer._should_skip_based_on_skip_criteria(request)
    assert result is True, "Should return True if a matching original_value_sha256 is found"


@pytest.mark.asyncio
async def test_send_prompt_async_exception_conv_id(mock_memory_instance, seed_prompt_group):
    prompt_target = MagicMock(PromptTarget)
    prompt_target.send_prompt_async = AsyncMock(side_effect=Exception("Test Exception"))

    normalizer = PromptNormalizer()

    with pytest.raises(Exception, match="Error sending prompt with conversation ID: 123"):
        await normalizer.send_prompt_async(
            seed_prompt_group=seed_prompt_group, target=prompt_target, conversation_id="123"
        )

    # Validate that first request is added to memory, then exception is added to memory
    assert (
        seed_prompt_group.prompts[0].value
        == mock_memory_instance.add_request_response_to_memory.call_args_list[0][1]["request"]
        .request_pieces[0]
        .original_value
    )
    assert (
        "Test Exception"
        in mock_memory_instance.add_request_response_to_memory.call_args_list[1][1]["request"]
        .request_pieces[0]
        .original_value
    )
