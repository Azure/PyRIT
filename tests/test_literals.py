# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import get_args, get_origin, Literal

from pyrit.models import PromptDataType, PromptResponseError, ChatMessageRole


def test_prompt_data_type():
    assert get_origin(PromptDataType) is Literal

    expected_literals = {"text", "image_path", "audio_path", "url", "error"}
    assert set(get_args(PromptDataType)) == expected_literals


def test_prompt_response_error():
    assert get_origin(PromptResponseError) is Literal

    expected_literals = {"blocked", "none", "processing", "unknown"}
    assert set(get_args(PromptResponseError)) == expected_literals


def test_chat_message_role():
    assert get_origin(ChatMessageRole) is Literal

    expected_literals = {"system", "user", "assistant"}
    assert set(get_args(ChatMessageRole)) == expected_literals
