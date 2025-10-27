# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import re

import pytest

from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import OpenAIResponseTarget


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "api_key", "model_name"),
    [
        (
            "PLATFORM_OPENAI_RESPONSES_ENDPOINT",
            "PLATFORM_OPENAI_RESPONSES_KEY",
            "PLATFORM_OPENAI_RESPONSES_MODEL",
        ),
        ("AZURE_OPENAI_RESPONSES_ENDPOINT", "AZURE_OPENAI_RESPONSES_KEY", "AZURE_OPENAI_RESPONSES_MODEL"),
    ],
)
async def test_aoai_responses_cfg(sqlite_instance, endpoint, api_key, model_name):
    lark_grammar = r"""
start: "I think that it is " SHORTTEXT
SHORTTEXT: /[^PpAaRrIiSs]{1,8}/
"""

    grammar_tool = {
        "type": "custom",
        "name": "CitiesGrammar",
        "description": "Constrains generation.",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": lark_grammar,
        },
    }

    args = {
        "endpoint": os.getenv(endpoint),
        "api_key": os.getenv(api_key),
        "model_name": os.getenv(model_name),
        "api_version": "2025-03-01-preview",
        "extra_body_parameters": {"tools": [grammar_tool], "tool_choice": "required"},
        "temperature": 1.0,
    }

    target = OpenAIResponseTarget(**args)

    message_piece = MessagePiece(
        role="user",
        original_value="What is the capital of France?",
        original_value_data_type="text",
    )
    prompt_request = Message(message_pieces=[message_piece])

    result = await target.send_prompt_async(prompt_request=prompt_request)

    assert len(result.message_pieces) == 2

    target_piece = result.message_pieces[1]
    assert target_piece.role == "assistant"
    response_text = target_piece.original_value
    assert response_text.startswith("I think that it is ")
    generation_text = response_text[len("I think that it is ") :]
    assert len(generation_text) <= 8
    assert re.match(r"[^PpAaRrIiSs]{1,8}", generation_text)
