# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import os
import uuid

import jsonschema
import pytest

from pyrit.models import MessagePiece
from pyrit.prompt_target import OpenAIResponseTarget


@pytest.fixture()
def gpt5_args():
    return {
        "endpoint": os.getenv("AZURE_OPENAI_GPT5_RESPONSES_ENDPOINT"),
        "model_name": os.getenv("AZURE_OPENAI_GPT5_MODEL"),
        # "api_key": os.getenv("AZURE_OPENAI_GPT5_KEY"),
        "use_entra_auth": True,
    }


@pytest.mark.asyncio
async def test_openai_responses_gpt5(sqlite_instance, gpt5_args):
    target = OpenAIResponseTarget(**gpt5_args)

    conv_id = str(uuid.uuid4())

    developer_piece = MessagePiece(
        role="developer",
        original_value="You are a helpful assistant.",
        original_value_data_type="text",
        conversation_id=conv_id,
        attack_identifier={"id": str(uuid.uuid4())},
    )
    sqlite_instance.add_message_to_memory(request=developer_piece.to_message())

    user_piece = MessagePiece(
        role="user",
        original_value="What is the capital of France?",
        original_value_data_type="text",
        conversation_id=conv_id,
    )

    result = await target.send_prompt_async(message=user_piece.to_message())
    assert result is not None
    assert len(result) == 1
    assert len(result[0].message_pieces) == 2
    assert result[0].message_pieces[0].role == "assistant"
    assert result[0].message_pieces[1].role == "assistant"
    # Hope that the model manages to give the correct answer somewhere (GPT-5 really should)
    assert "Paris" in result.message_pieces[1].converted_value


@pytest.mark.asyncio
async def test_openai_responses_gpt5_json_schema(sqlite_instance, gpt5_args):
    target = OpenAIResponseTarget(**gpt5_args)

    conv_id = str(uuid.uuid4())

    developer_piece = MessagePiece(
        role="developer",
        original_value="You are an expert in the lore of cats.",
        original_value_data_type="text",
        conversation_id=conv_id,
        attack_identifier={"id": str(uuid.uuid4())},
    )
    sqlite_instance.add_message_to_memory(request=developer_piece.to_message())

    cat_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 12},
            "age": {"type": "integer", "minimum": 0, "maximum": 20},
            "colour": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 255},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "required": ["name", "age", "colour"],
        "additionalProperties": False,
    }

    prompt = "Create a JSON object that describes a mystical cat "
    prompt += "with the following properties: name, age, colour."

    user_piece = MessagePiece(
        role="user",
        original_value=prompt,
        original_value_data_type="text",
        conversation_id=conv_id,
        prompt_metadata={"response_format": "json", "json_schema": json.dumps(cat_schema)},
    )

    response = await target.send_prompt_async(message=user_piece.to_message())

    response_content = response.get_value(1)
    response_json = json.loads(response_content)
    jsonschema.validate(instance=response_json, schema=cat_schema)


@pytest.mark.asyncio
async def test_openai_responses_gpt5_json_object(sqlite_instance, gpt5_args):
    target = OpenAIResponseTarget(**gpt5_args)

    conv_id = str(uuid.uuid4())

    developer_piece = MessagePiece(
        role="developer",
        original_value="You are an expert in the lore of cats.",
        original_value_data_type="text",
        conversation_id=conv_id,
        attack_identifier={"id": str(uuid.uuid4())},
    )

    sqlite_instance.add_message_to_memory(request=developer_piece.to_message())

    prompt = "Create a JSON object that describes a mystical cat "
    prompt += "with the following properties: name, age, colour."

    user_piece = MessagePiece(
        role="user",
        original_value=prompt,
        original_value_data_type="text",
        conversation_id=conv_id,
        prompt_metadata={"response_format": "json"},
    )
    response = await target.send_prompt_async(message=user_piece.to_message())

    response_content = response.get_value(1)
    response_json = json.loads(response_content)
    assert response_json is not None
    # Can't assert more, since the failure could be due to a bad generation by the model
