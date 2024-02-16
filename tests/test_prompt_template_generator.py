# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.common.prompt_template_generator import PromptTemplateGenerator
from pyrit.models import ChatMessage


@pytest.fixture
def prompt_template_generator() -> PromptTemplateGenerator:
    prompt_template_generator = PromptTemplateGenerator()
    return prompt_template_generator


def test_generate_template_requires_messages(
    prompt_template_generator: PromptTemplateGenerator,
):
    messages: list[ChatMessage] = []
    with pytest.raises(ValueError) as e:
        prompt_template_generator.generate_template(messages)
    assert str(e.value) == "The messages list cannot be empty."


def test_generate_default_template_first_call_requires_two_messages(
    prompt_template_generator: PromptTemplateGenerator,
):
    with pytest.raises(ValueError) as e:
        prompt_template_generator.generate_template([ChatMessage(role="user", content="content1")])
    assert str(e.value) == "At least two chat message objects are required for the first call. Obtained only 1."


def test_generate_default_template_first_call_success(
    prompt_template_generator: PromptTemplateGenerator,
):
    prompt_template = prompt_template_generator.generate_template(
        [
            ChatMessage(role="system", content="system content"),
            ChatMessage(role="user", content="user content"),
        ]
    )
    assert prompt_template == "SYSTEM:system contentUSER:user contentASSISTANT:"


def test_generate_template_subsequent_call(
    prompt_template_generator: PromptTemplateGenerator,
):
    initial_messages = [
        ChatMessage(role="system", content="system content"),
        ChatMessage(role="user", content="user content"),
    ]
    prompt_template = prompt_template_generator.generate_template(initial_messages)
    assert prompt_template == "SYSTEM:system contentUSER:user contentASSISTANT:"
    chat_messages_with_history = [
        ChatMessage(role="system", content="system content"),
        ChatMessage(role="user", content="user content1"),
        ChatMessage(role="assistant", content="assistant1"),
        ChatMessage(role="user", content="user content2"),
    ]
    final_prompt_template = prompt_template_generator.generate_template(chat_messages_with_history)
    expected_template = "SYSTEM:system contentUSER:user contentASSISTANT:assistant1USER:user content2ASSISTANT:"
    assert final_prompt_template == expected_template
