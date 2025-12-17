# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import MockPromptTarget, get_image_message_piece

from pyrit.exceptions import EmptyResponseException
from pyrit.memory import CentralMemory
from pyrit.models import (
    Message,
    MessagePiece,
    PromptDataType,
    SeedGroup,
    SeedPrompt,
)
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
def response() -> Message:
    conversation_id = "123"
    image_message_piece = get_image_message_piece()
    image_message_piece.role = "assistant"
    image_message_piece.conversation_id = conversation_id
    return Message(
        message_pieces=[
            MessagePiece(role="assistant", original_value="Hello", conversation_id=conversation_id),
            MessagePiece(role="assistant", original_value="part 2", conversation_id=conversation_id),
            image_message_piece,
        ]
    )


@pytest.fixture
def seed_group() -> SeedGroup:
    return SeedGroup(
        seeds=[
            SeedPrompt(
                value="Hello",
                data_type="text",
                role="system",
                sequence=1,
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


def assert_message_piece_hashes_set(request: Message):
    assert request
    assert request.message_pieces
    for piece in request.message_pieces:
        assert piece.original_value_sha256
        assert piece.converted_value_sha256


@pytest.mark.asyncio
async def test_send_prompt_async_multiple_converters(mock_memory_instance, seed_group):
    prompt_target = MockPromptTarget()
    request_converters = [
        PromptConverterConfiguration(converters=[Base64Converter(), StringJoinConverter(join_value="_")])
    ]

    normalizer = PromptNormalizer()

    await normalizer.send_prompt_async(
        seed_group=seed_group, request_converter_configurations=request_converters, target=prompt_target
    )

    assert prompt_target.prompt_sent == ["S_G_V_s_b_G_8_="]


@pytest.mark.asyncio
async def test_send_prompt_async_no_response_adds_memory(mock_memory_instance, seed_group):
    prompt_target = AsyncMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=None)

    normalizer = PromptNormalizer()

    await normalizer.send_prompt_async(seed_group=seed_group, target=prompt_target)
    assert mock_memory_instance.add_message_to_memory.call_count == 1

    request = mock_memory_instance.add_message_to_memory.call_args[1]["request"]
    assert_message_piece_hashes_set(request)


@pytest.mark.asyncio
async def test_send_prompt_async_empty_response_exception_handled(mock_memory_instance, seed_group):
    # Use MagicMock with send_prompt_async as AsyncMock to avoid coroutine warnings on other methods
    prompt_target = MagicMock()
    prompt_target.send_prompt_async = AsyncMock(side_effect=EmptyResponseException(message="Empty response"))

    normalizer = PromptNormalizer()

    response = await normalizer.send_prompt_async(seed_group=seed_group, target=prompt_target)

    assert mock_memory_instance.add_message_to_memory.call_count == 2

    assert response.message_pieces[0].response_error == "empty"
    assert response.message_pieces[0].original_value == ""
    assert response.message_pieces[0].original_value_data_type == "text"

    assert_message_piece_hashes_set(response)


@pytest.mark.asyncio
async def test_send_prompt_async_request_response_added_to_memory(mock_memory_instance, seed_group):
    # Use MagicMock with send_prompt_async as AsyncMock to avoid coroutine warnings
    prompt_target = MagicMock()

    response = MessagePiece(role="assistant", original_value="test_response").to_message()

    prompt_target.send_prompt_async = AsyncMock(return_value=[response])

    normalizer = PromptNormalizer()

    await normalizer.send_prompt_async(seed_group=seed_group, target=prompt_target)

    assert mock_memory_instance.add_message_to_memory.call_count == 2

    seed_prompt_value = seed_group.prompts[0].value
    # Validate that first request is added to memory, then response is added to memory
    assert (
        seed_prompt_value
        == mock_memory_instance.add_message_to_memory.call_args_list[0][1]["request"].message_pieces[0].original_value
    )
    assert (
        "test_response"
        == mock_memory_instance.add_message_to_memory.call_args_list[1][1]["request"].message_pieces[0].original_value
    )

    assert mock_memory_instance.add_message_to_memory.call_args_list[1].called_after(prompt_target.send_prompt_async)


@pytest.mark.asyncio
async def test_send_prompt_async_exception(mock_memory_instance, seed_group):
    prompt_target = AsyncMock()

    seed_prompt_value = seed_group.prompts[0].value

    normalizer = PromptNormalizer()

    with patch("pyrit.models.construct_response_from_request") as mock_construct:
        mock_construct.return_value = "test"

        try:
            await normalizer.send_prompt_async(seed_group=seed_group, target=prompt_target)
        except ValueError:
            assert mock_memory_instance.add_message_to_memory.call_count == 2

            # Validate that first request is added to memory, then exception is added to memory
            assert (
                seed_prompt_value
                == mock_memory_instance.add_message_to_memory.call_args_list[0][1]["request"]
                .message_pieces[0]
                .original_value
            )
            assert (
                "test_exception"
                == mock_memory_instance.add_message_to_memory.call_args_list[1][1]["request"]
                .message_pieces[0]
                .original_value
            )


@pytest.mark.asyncio
async def test_send_prompt_async_empty_exception(mock_memory_instance, seed_group):
    prompt_target = AsyncMock()
    prompt_target.send_prompt_async = AsyncMock(side_effect=Exception(""))

    normalizer = PromptNormalizer()

    with pytest.raises(Exception, match="Error sending prompt with conversation ID"):
        await normalizer.send_prompt_async(seed_group=seed_group, target=prompt_target)


@pytest.mark.asyncio
async def test_send_prompt_async_different_sequences(mock_memory_instance):
    """Test that sending prompts with different sequences raises ValueError."""
    prompt_target = AsyncMock()
    normalizer = PromptNormalizer()

    prompts = [
        SeedPrompt(value="test1", sequence=1, role="user"),
        SeedPrompt(value="test2", sequence=2, role="user"),
    ]  # Different sequence
    group = SeedGroup(seeds=prompts)

    with pytest.raises(ValueError, match="All SeedPrompts in the SeedGroup must have the same sequence"):
        await normalizer.send_prompt_async(seed_group=group, target=prompt_target)


@pytest.mark.asyncio
async def test_send_prompt_async_mixed_sequence_types(mock_memory_instance):
    """Test that sending prompts with mixed sequence types (None and int) raises ValueError."""
    prompt_target = AsyncMock()
    normalizer = PromptNormalizer()

    prompts = [
        SeedPrompt(value="test1", sequence=1, role="user"),
        SeedPrompt(value="test2", role="user"),
    ]  # No sequence (will default to None)
    group = SeedGroup(seeds=prompts)

    with pytest.raises(ValueError, match="All SeedPrompts in the SeedGroup must have the same sequence"):
        await normalizer.send_prompt_async(seed_group=group, target=prompt_target)


@pytest.mark.asyncio
async def test_send_prompt_async_adds_memory_twice(mock_memory_instance, seed_group, response: Message):
    prompt_target = MagicMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=[response])

    normalizer = PromptNormalizer()

    response = await normalizer.send_prompt_async(seed_group=seed_group, target=prompt_target)
    assert mock_memory_instance.add_message_to_memory.call_count == 2


@pytest.mark.asyncio
async def test_send_prompt_async_no_converters_response(mock_memory_instance, seed_group, response: Message):

    prompt_target = MagicMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=[response])

    normalizer = PromptNormalizer()

    # Send prompt async and check the response
    response = await normalizer.send_prompt_async(seed_group=seed_group, target=prompt_target)
    assert response.get_value() == "Hello", "There were no response converters"


@pytest.mark.asyncio
async def test_send_prompt_async_converters_response(mock_memory_instance, seed_group, response: Message):

    prompt_target = MagicMock()
    prompt_target.send_prompt_async = AsyncMock(return_value=[response])

    response_converter = PromptConverterConfiguration(converters=[Base64Converter()], indexes_to_apply=[0])

    normalizer = PromptNormalizer()

    response = await normalizer.send_prompt_async(
        seed_group=seed_group,
        response_converter_configurations=[response_converter],
        target=prompt_target,
    )

    assert response.get_value() == "SGVsbG8="


@pytest.mark.asyncio
async def test_send_prompt_async_image_converter(mock_memory_instance):
    prompt_target = MagicMock(PromptTarget)
    prompt_target.send_prompt_async = AsyncMock(
        return_value=[MessagePiece(role="assistant", original_value="response").to_message()]
    )

    mock_image_converter = MagicMock(PromptConverter)

    filename = ""

    with tempfile.NamedTemporaryFile(delete=False) as f:
        filename = f.name
        f.write(b"Hello")

        mock_image_converter.convert_tokens_async = AsyncMock(
            return_value=ConverterResult(
                output_type="image_path",
                output_text=filename,
            )
        )

        prompt_converters = PromptConverterConfiguration(converters=[mock_image_converter])

        prompt_text = "Hello"

        seed_group = SeedGroup(seeds=[SeedPrompt(value=prompt_text, data_type="text")])

        normalizer = PromptNormalizer()
        # Mock the async read_file method
        normalizer._memory.results_storage_io.read_file = AsyncMock(return_value=b"mocked data")

        response = await normalizer.send_prompt_async(
            seed_group=seed_group,
            target=prompt_target,
            request_converter_configurations=[prompt_converters],
        )

        # verify the prompt target received the correct arguments from the normalizer
        sent_request = prompt_target.send_prompt_async.call_args.kwargs["message"].message_pieces[0]
        assert sent_request.converted_value == filename
        assert sent_request.converted_value_data_type == "image_path"

        assert_message_piece_hashes_set(response)
    os.remove(filename)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_requests_per_minute", [None, 10])
@pytest.mark.parametrize("batch_size", [1, 10])
async def test_prompt_normalizer_send_prompt_batch_async_throws(
    mock_memory_instance, seed_group, max_requests_per_minute, batch_size
):
    prompt_target = MockPromptTarget(rpm=max_requests_per_minute)

    request_converters = PromptConverterConfiguration(
        converters=[Base64Converter(), StringJoinConverter(join_value="_")]
    )

    normalizer_request = NormalizerRequest(
        seed_group=seed_group,
        request_converter_configurations=[request_converters],
    )

    normalizer = PromptNormalizer()

    # Mock asyncio.sleep to avoid 6s delay in rate limiting test
    with patch("asyncio.sleep", new_callable=AsyncMock):
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
async def test_build_message(mock_memory_instance, seed_group):

    labels = {"label1": "value1", "label2": "value2"}

    conversation_id = uuid.uuid4()

    prompt_target = MockPromptTarget()
    request_converters = [
        PromptConverterConfiguration(converters=[Base64Converter(), StringJoinConverter(join_value="_")])
    ]

    normalizer = PromptNormalizer()

    response = await normalizer._build_message(
        seed_group=seed_group,
        conversation_id=conversation_id,
        request_converter_configurations=request_converters,
        target=prompt_target,
        labels=labels,
    )

    # Check all message pieces in the response have the same conversation ID
    assert len(set(message_piece.conversation_id for message_piece in response.message_pieces)) == 1

    assert response.message_pieces[0].sequence == 1
    assert len(set(message_piece.sequence for message_piece in response.message_pieces)) == 1

    assert response.message_pieces[0].role == "system"
    assert len(set(message_piece.role for message_piece in response.message_pieces)) == 1

    # Check sequence is set correctly
    assert len(set(message_piece.sequence for message_piece in response.message_pieces)) == 1


@pytest.mark.asyncio
async def test_convert_response_values_index(mock_memory_instance, response: Message):
    response_converter = PromptConverterConfiguration(converters=[Base64Converter()], indexes_to_apply=[0])

    normalizer = PromptNormalizer()

    await normalizer.convert_values(converter_configurations=[response_converter], message=response)
    assert response.get_value() == "SGVsbG8=", "Converter should be applied here"
    assert response.get_value(1) == "part 2", "Converter should not be applied since we specified only 0"


@pytest.mark.asyncio
async def test_convert_response_values_type(mock_memory_instance, response: Message):
    response_converter = PromptConverterConfiguration(
        converters=[Base64Converter()], prompt_data_types_to_apply=["text"]
    )

    normalizer = PromptNormalizer()

    await normalizer.convert_values(converter_configurations=[response_converter], message=response)
    assert response.get_value() == "SGVsbG8="
    assert response.get_value(1) == "cGFydCAy"


@pytest.mark.asyncio
async def test_send_prompt_async_exception_conv_id(mock_memory_instance, seed_group):
    prompt_target = MagicMock(PromptTarget)
    prompt_target.send_prompt_async = AsyncMock(side_effect=Exception("Test Exception"))

    normalizer = PromptNormalizer()

    with pytest.raises(Exception, match="Error sending prompt with conversation ID: 123"):
        await normalizer.send_prompt_async(seed_group=seed_group, target=prompt_target, conversation_id="123")

    # Validate that first request is added to memory, then exception is added to memory
    assert (
        seed_group.prompts[0].value
        == mock_memory_instance.add_message_to_memory.call_args_list[0][1]["request"].message_pieces[0].original_value
    )
    assert (
        "Test Exception"
        in mock_memory_instance.add_message_to_memory.call_args_list[1][1]["request"].message_pieces[0].original_value
    )
