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


def test_prompt_init_valid_arguments():
    prompt_target = MockPromptTarget()
    prompt_converters = [MockPromptConverter()]
    prompt_text = "Hello"
    conversation_id = "123"

    prompt = PromptRequestPiece(
        prompt_target=prompt_target,
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        conversation_id=conversation_id,
    )

    assert prompt._prompt_target == prompt_target
    assert prompt._prompt_converters == prompt_converters
    assert prompt._prompt_text == prompt_text
    assert prompt.conversation_id == conversation_id


def test_prompt_init_invalid_prompt_target():
    prompt_target = "InvalidPromptTarget"
    prompt_converters = [MockPromptConverter()]
    prompt_text = "Hello"
    conversation_id = "123"

    with pytest.raises(ValueError):
        PromptRequestPiece(
            prompt_target=prompt_target,
            prompt_converters=prompt_converters,
            prompt_text=prompt_text,
            conversation_id=conversation_id,
        )


def test_prompt_init_invalid_prompt_converters():
    prompt_target = MockPromptTarget()
    prompt_converters = ["InvalidPromptConverter"]
    prompt_text = "Hello"
    conversation_id = "123"

    with pytest.raises(ValueError):
        PromptRequestPiece(
            prompt_target=prompt_target,
            prompt_converters=prompt_converters,
            prompt_text=prompt_text,
            conversation_id=conversation_id,
        )


def test_prompt_init_empty_prompt_converters():
    prompt_target = MockPromptTarget()
    prompt_converters = []
    prompt_text = "Hello"
    conversation_id = "123"

    with pytest.raises(ValueError):
        PromptRequestPiece(
            prompt_target=prompt_target,
            prompt_converters=prompt_converters,
            prompt_text=prompt_text,
            conversation_id=conversation_id,
        )


def test_prompt_init_invalid_prompt_text():
    prompt_target = MockPromptTarget()
    prompt_converters = [MockPromptConverter()]
    prompt_text = 123
    conversation_id = "123"

    with pytest.raises(ValueError):
        PromptRequestPiece(
            prompt_target=prompt_target,
            prompt_converters=prompt_converters,
            prompt_text=prompt_text,
            conversation_id=conversation_id,
        )


def test_prompt_init_invalid_conversation_id():
    prompt_target = MockPromptTarget()
    prompt_converters = [MockPromptConverter()]
    prompt_text = "Hello"
    conversation_id = 123

    with pytest.raises(ValueError):
        PromptRequestPiece(
            prompt_target=prompt_target,
            prompt_converters=prompt_converters,
            prompt_text=prompt_text,
            conversation_id=conversation_id,
        )


def test_send_prompt_multiple_converters():
    prompt_target = MockPromptTarget()
    prompt_converters = [Base64Converter(), StringJoinConverter(join_value="_")]
    prompt_text = "Hello"
    conversation_id = "123"

    prompt = PromptRequestPiece(
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

    prompt = PromptRequestPiece(
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

    prompt = PromptRequestPiece(
        prompt_target=prompt_target,
        prompt_converters=prompt_converters,
        prompt_text=prompt_text,
        conversation_id=conversation_id,
    )

    normalizer = PromptNormalizer(memory=None)

    await normalizer.send_prompt_batch_async([prompt])
    assert prompt_target.prompt_sent == ["S_G_V_s_b_G_8_="]
