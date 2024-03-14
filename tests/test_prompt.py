# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_converter import Base64Converter, StringJoinConverter
from pyrit.prompt_normalizer import Prompt, PromptNormalizer
from pyrit.prompt_converter import PromptConverter

from tests.mocks import MockPromptTarget


class MockPromptConverter(PromptConverter):

    def __init__(self) -> None:
        pass

    def convert(self, prompts: list[str]) -> list[str]:
        return prompts

    def is_one_to_one_converter(self) -> bool:
        return True


def test_prompt_init_valid_arguments():
    prompt_target = MockPromptTarget()
    prompt_converters = [MockPromptConverter()]
    prompt_text = "Hello"
    conversation_id = "123"

    prompt = Prompt(
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
        Prompt(
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
        Prompt(
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
        Prompt(
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
        Prompt(
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
        Prompt(
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
async def test_prompt_normalizyer_send_prompt_batch_async():
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
