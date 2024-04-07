# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.memory.memory_models import PromptDataType
from pyrit.prompt_converter import Base64Converter, StringJoinConverter
from pyrit.prompt_normalizer import PromptRequestPiece, PromptNormalizer
from pyrit.prompt_converter import PromptConverter

from tests.mocks import MockPromptTarget


class MockPromptConverter(PromptConverter):

    def __init__(self) -> None:
        pass

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> str:
        return prompt

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"


def test_prompt_request_piece_init_valid_arguments():
    prompt_converters = [MockPromptConverter()]
    prompt_text = "Hello"
    metadata="meta"

    prompt = PromptRequestPiece(
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        prompt_data_type="text",
        metadata=metadata,
    )

    assert prompt.prompt_converters == prompt_converters
    assert prompt.prompt_text == prompt_text
    assert prompt.prompt_data_type == "text"
    assert prompt.metadata == metadata

def test_prompt_init_no_metadata():
    prompt_converters = [MockPromptConverter()]
    prompt_text = "Hello"

    prompt = PromptRequestPiece(
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        prompt_data_type="text",
    )

    assert prompt.prompt_converters == prompt_converters
    assert prompt.prompt_text == prompt_text
    assert prompt.prompt_data_type == "text"
    assert not prompt.metadata

def test_prompt_request_piece_init_invalid_converter():
    prompt_text = "Hello"
    metadata="meta"

    with pytest.raises(ValueError):
        PromptRequestPiece(
            prompt_converters=["InvalidPromptConverter"],
            prompt_text=prompt_text,
            prompt_data_type="text",
            metadata=metadata,
        )

def test_prompt_init_empty_prompt_converters():
    prompt_text = "Hello"
    metadata="meta"

    with pytest.raises(ValueError):
        PromptRequestPiece(
            prompt_converters=[],
            prompt_text=prompt_text,
            prompt_data_type="text",
            metadata=metadata,
        )



def test_prompt_init_invalid_prompt_text():
    metadata="meta"

    with pytest.raises(ValueError):
        PromptRequestPiece(
            prompt_converters=[],
            prompt_text=123,
            prompt_data_type="text",
            metadata=metadata,
        )
        

def test_send_prompt_multiple_converters():
    prompt_target = MockPromptTarget()
    prompt_converters = [Base64Converter(), StringJoinConverter(join_value="_")]
    prompt_text = "Hello"
    conversation_id = "123"

    prompt = Prompt(
        prompt_target=prompt_target,
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        conversation_id=conversation_id,
    )

    normalizer_id = "456"
    prompt.send_prompt(normalizer_id=normalizer_id)

    assert prompt_target.prompt_sent == ["S_G_V_s_b_G_8_="]


@pytest.mark.asyncio
async def test_send_prompt_async_multiple_converters():
    prompt_target = MockPromptTarget()
    prompt_converters = [Base64Converter(), StringJoinConverter(join_value="_")]
    prompt_text = "Hello"
    conversation_id = "123"

    prompt = Prompt(
        prompt_target=prompt_target,
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        conversation_id=conversation_id,
    )

    normalizer_id = "456"
    await prompt.send_prompt_async(normalizer_id=normalizer_id)

    assert prompt_target.prompt_sent == ["S_G_V_s_b_G_8_="]


@pytest.mark.asyncio
async def test_prompt_normalizer_send_prompt_batch_async():
    prompt_target = MockPromptTarget()
    prompt_converters = [Base64Converter(), StringJoinConverter(join_value="_")]
    prompt_text = "Hello"
    conversation_id = "123"

    prompt = Prompt(
        prompt_target=prompt_target,
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        conversation_id=conversation_id,
    )

    normalizer = PromptNormalizer(memory=None)

    await normalizer.send_prompt_batch_async([prompt])
    assert prompt_target.prompt_sent == ["S_G_V_s_b_G_8_="]
