# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

import pytest

from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import OpenAIResponseTarget


@pytest.mark.asyncio
async def test_aoai_responses_cfg(sqlite_instance):
    endpoint = os.getenv("AZURE_OPENAI_RESPONSES_ENDPOINT")
    model_name = os.getenv("AZURE_OPENAI_RESPONSES_MODEL")
    api_version = os.getenv("AZURE_OPENAI_RESPONSES_API_VERSION")

    lark_grammar = r"""
start: "I think that it is " SHORTTEXT 
SHORTTEXT: /[^Pp]{1,8}/
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

    target = OpenAIResponseTarget(
        endpoint=endpoint,
        model_name=model_name,
        use_entra_auth=True,
        api_version=api_version,
        extra_body_parameters={"tools": [grammar_tool], "tool_choice": "required"},
        temperature=1.0,
    )

    message_piece = MessagePiece(
        role="user",
        original_value="What is the capital of France?",
        original_value_data_type="text",
    )
    prompt_request = Message(message_pieces=[message_piece])

    result = await target.send_prompt_async(prompt_request=prompt_request)

    for idx, piece in enumerate(result.message_pieces):
        print(f"{idx} | {piece.role}: {piece.original_value}")

    assert False, "Force failure for output inspection"
